import subprocess

def update_requirements(mode=""):
    if mode == "pipreqs":
        try:
            import pipreqs
            subprocess.call("pipreqs --force .", shell=True)
        except ModuleNotFoundError:
            print("Please run pip install pipreqs! Requirements generated with standard pip freeze..")
            subprocess.call("pip freeze > requirements.txt", shell=True)

    else:
        subprocess.call("pip freeze > requirements.txt", shell=True)




if __name__ == '__main__':
    update_requirements("")