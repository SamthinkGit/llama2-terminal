import io
import os
import sys
import inquirer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

system_choices = [
    inquirer.List(
        'system',
        message="Select system to use",
        choices=['powershell.exe', 'cmd.exe', 'python.exe'],
    )
]

system_shortnames = {
    'powershell.exe': 'PS',
    'cmd.exe': 'CMD',
    'python.exe': 'Py'
}

system_end_msgs = {
    'powershell.exe': '##END_OF_OUTPUT##',
    'cmd.exe': 'REM END_OF_OUTPUT',
    'python.exe': '##END_OF_OUTPUT##'
}

system_pwd = {
    'powershell.exe': 'pwd | findstr :',
    'cmd.exe': 'cd',
}

def get_l2t_path():
    
        l2t_path = os.environ.get("L2T_PATH")
        if not l2t_path:
            raise EnvironmentError("Environment variable 'L2T_PATH' is not defined.")
        return l2t_path
