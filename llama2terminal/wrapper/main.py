from llama2terminal.wrapper.terminal import ShellWrapper

if __name__ == '__main__':
    app = ShellWrapper()
    app.cmdloop()