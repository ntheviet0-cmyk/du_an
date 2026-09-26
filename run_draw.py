import subprocess, sys, os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draw_graph_arch.py")
result = subprocess.run([sys.executable, path], capture_output=True, text=True, encoding='utf-8')
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
