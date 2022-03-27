from apps.process_onefile import * 
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Microservice B: Pifu'

@app.route('/<filename>', methods=['GET'])
def run_pifu(filename):
    print(f"Running pifu pipeline on image {filename}")
    do(filename,opt,netMR,cuda)
    return f"Finish processing on image {filename}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=CONFIG['pifu_port'],debug=True)