import subprocess
import threading
import config
from queue import Queue

class CommandLineReader:
    
    def __init__(self, sys_type):
        self.sys_type = sys_type
        self.stderr_queue = Queue()
        self._end_output_signal = config.system_end_msgs[sys_type]
        self.start()

    def start(self):
        self.shell_process = subprocess.Popen([self.sys_type], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.stderr_thread = threading.Thread(target=self.error_reader)
        self.stop_event = threading.Event()
        self.stderr_thread.start()

        # Run a command for fully starting the process
        self.run_command("")

    def error_reader(self):
        while not self.stop_event.is_set():
            line = self.shell_process.stderr.readline()
            if line.strip():
                self.stderr_queue.put(line)

    def run_command(self, cmd):

        # Send the command to the process
        delimiter = self._end_output_signal
        self.shell_process.stdin.write(cmd + '\n')
        self.shell_process.stdin.write(delimiter + '\n')
        self.shell_process.stdin.flush()

        # Remove the query line\s
        line = ""
        while cmd not in line:
            line = self.shell_process.stdout.readline()

        # Read stdout until delimeter 
        output_lines = []
        while True:
            line = self.shell_process.stdout.readline()
            if not line:
                continue
            if delimiter in line:
                break
            if line.strip():
                output_lines.append(line) 

        # Read non-blocking errors
        error_lines = []
        while not self.stderr_queue.empty():
            line = self.stderr_queue.get_nowait()
            if line.strip():
                error_lines.append(line)

        # Flush other outputs 
        self.shell_process.stdout.flush()
        self.shell_process.stderr.flush()

        return ''.join(output_lines), ''.join(error_lines)

    def close(self):
        if self.shell_process:
            self.stop_event.set()
            self.shell_process.terminate()
            self.stderr_thread.join()
