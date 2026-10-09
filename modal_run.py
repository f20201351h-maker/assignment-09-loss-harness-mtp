"""Execute the loss-harness notebook top to bottom on a Modal T4 and save the executed copy locally.

    modal run modal_run.py                      # full run: overwrites the notebook, results/ and figures/ here
    modal run modal_run.py --quick --out tmp    # smoke test (tiny data, few steps) into ./tmp

The notebook itself is an ordinary .ipynb; Modal is only the machine it runs on. The container gets no
secrets: the data is a public Hugging Face dataset, downloaded anonymously.
"""
import io
import pathlib
import tarfile

import modal

HERE = pathlib.Path(__file__).parent
NOTEBOOK = "loss_harness.ipynb"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "numpy==2.1.3", "pandas==2.2.3", "matplotlib==3.9.2", "jinja2==3.1.4",
    "tiktoken==0.8.0", "datasets==3.2.0", "nbformat==5.10.4", "nbclient==0.10.2", "ipykernel==6.29.5",
)
app = modal.App("loss-harness", image=image)


@app.function(gpu="T4", cpu=4.0, memory=16384, timeout=75 * 60)
def execute(nb_json: str, quick: bool) -> tuple[bytes, str]:
    import os
    import traceback

    import nbformat
    from nbclient import NotebookClient

    work = pathlib.Path("/root/work")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    os.environ["S9_QUICK"] = "1" if quick else "0"
    nb = nbformat.reads(nb_json, as_version=4)
    error = ""
    try:
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(work)}}).execute()
    except Exception:
        error = traceback.format_exc()[-4000:]
    nbformat.write(nb, work / NOTEBOOK)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in (NOTEBOOK, "results", "figures"):
            if (work / name).exists():
                tar.add(work / name, arcname=name)
    return buf.getvalue(), error


@app.local_entrypoint()
def main(quick: bool = False, out: str = "."):
    out_dir = (HERE / out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_json = (HERE / NOTEBOOK).read_text(encoding="utf-8")
    payload, error = execute.remote(nb_json, quick)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        tar.extractall(out_dir)
    print(f"executed notebook and artifacts written to {out_dir}")
    if error:
        print("NOTEBOOK EXECUTION FAILED:\n" + error)
        raise SystemExit(1)
