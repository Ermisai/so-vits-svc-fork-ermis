from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="so-vits-svc-fork-ermis",
    version="0.1.0",
    author="Vittor Lima",
    author_email="vittorlima@discente.ufg.br",
    description="So-VITS-SVC Fork - Módulo de inferência isolada para conversão de voz",
    long_description=open("README.md", encoding="utf-8").read(),
    url="https://github.com/Ermisai/so-vits-svc-fork-ermis",
    packages=["so_vits_svc_fork_ermis", "so_vits_svc_fork_ermis.inference", "so_vits_svc_fork_ermis.modules", "so_vits_svc_fork_ermis.modules.decoders", "so_vits_svc_fork_ermis.modules.decoders.hifigan", "so_vits_svc_fork_ermis.modules.decoders.mb_istft"],
    package_dir={"so_vits_svc_fork_ermis": "."},
    classifiers=[],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)
