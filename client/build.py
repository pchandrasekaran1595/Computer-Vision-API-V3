import PyInstaller.__main__

PyInstaller.__main__.run([
    "main.py",
    "--clean",
    "--onefile",
    "--console",
    "--distpath",
    "./",
    "--name",
    "cv-api-v3-client",
    "--icon",
    "icon.ico",
    "--log-level",
    "DEBUG",
])