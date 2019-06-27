import os

for i in range(500):
    cmd = "./ftrl_tool_main -c ftrl_client.conf -f Feedback -u 13214 -a "+str(i)+" -r 1"
    os.system(cmd)