
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    """Ana sohbet sayfasını döndürür."""
    return render_template("index.html")


if __name__ == "__main__":
    # debug=True yalnızca geliştirme ortamı içindir,
    # canlı (production) ortamda kapatılmalıdır.
   app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)