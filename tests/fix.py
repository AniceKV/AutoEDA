import re

with open('tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('plt.close("all")', 'plt.close()')

with open('tools.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
