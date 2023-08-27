import time

def typing_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)