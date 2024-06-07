import webbrowser
import os

# Đường dẫn tới tệp HTML
html_file = 'gold.html'

# Mở tệp HTML trong trình duyệt mặc định
webbrowser.open('file://' + os.path.realpath(html_file))
