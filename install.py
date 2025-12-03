import subprocess
import sys


CONDA_PREFIX = ["conda", "run", "-n", "bayesrl"]


def create_conda_env():
    print(f"📦 Creating conda env [bayesrl] ...")
    subprocess.run(
        ["conda", "create", "--name", "bayesrl", "python<3.12", "-y"],
        check=True
    )
    print(f"✅ Conda env [bayesrl] created.\n")


def has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is available using `nvidia-smi`."""
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def detect_cuda_version() -> str | None:
    """Try to detect CUDA version using `nvcc --version`."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if "release" in line:
                version = line.split("release")[-1].split(",")[0].strip()
                major = version.split(".")[0]
                return major  # e.g., '12' or '11'
    except Exception:
        return None


def install_jax():
    if has_nvidia_gpu():
        cuda_major = detect_cuda_version()
        if cuda_major == "12":
            pkg = "jax[cuda12]"
            repo = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        elif cuda_major == "11":
            pkg = "jax[cuda11]"
            repo = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    else:
        print("⚠️ CUDA version not detected; defaulting to CPU JAX.")
        pkg = "jax"
        repo = None

    print(f"📦 Installing {pkg} ...")
    cmd = ["pip", "install", "-U", pkg]
    if repo:
        cmd += ["-f", repo]
    subprocess.run(
        CONDA_PREFIX + cmd,
        check=True
    )
    print("✅ Jax installation completed.\n")


def install_requirements():
    print(f"📦 Installing requirements ...")
    subprocess.run(
        CONDA_PREFIX + ["pip", "install", "-r", "requirements.txt"],
        check=True
    )
    print("✅ Requirements installation completed.\n")


if __name__ == "__main__":
    create_conda_env()
    install_jax()
    install_requirements()
    print("🔥 Everything is set; please activate [bayesrl] for use.")
