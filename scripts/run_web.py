"""Run the minimal CoastLearn upload interface."""

from __future__ import annotations

import os

from coastlearn.webapp import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("COASTLEARN_HOST", "127.0.0.1"),
        port=int(os.environ.get("COASTLEARN_PORT", "5000")),
        debug=False,
    )
