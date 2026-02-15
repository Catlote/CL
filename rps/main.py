from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Tiny 10-neuron NN
input_size = 6
hidden_size = 10
output_size = 3

W1 = np.random.randn(hidden_size, input_size) * 0.1
b1 = np.zeros((hidden_size,1))
W2 = np.random.randn(output_size, hidden_size) * 0.1
b2 = np.zeros((output_size,1))
learning_rate = 0.05
history = []

# plain text moves
moves = ["rock","paper","scissors"]

def one_hot(move):
    mapping = {
        "rock": [1,0,0],
        "paper": [0,1,0],
        "scissors": [0,0,1]
    }
    return np.array(mapping[move]).reshape(3,1)

def encode_input(history, N=2):
    x = np.zeros((input_size,1))
    for i in range(1, min(len(history)+1,N+1)):
        x[3*(N-i):3*(N-i+1),0] = one_hot(history[-i]).flatten()
    return x

def relu(x): return np.maximum(0,x)
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=0)

def forward(x):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = softmax(z2)
    return a1, a2

def backward(x, a1, a2, y_true):
    global W1,b1,W2,b2
    dz2 = a2 - y_true
    dW2 = dz2 @ a1.T
    db2 = dz2
    da1 = W2.T @ dz2
    dz1 = da1 * (a1>0)
    dW1 = dz1 @ x.T
    db1 = dz1
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/play")
async def play(data: dict):
    global history
    human_move = data["move"]
    history.append(human_move)

    x = encode_input(history)
    a1, a2 = forward(x)
    neurons = a1.flatten().tolist()

    predicted_human = np.argmax(a2)
    beats = [1,2,0]  # rock->paper, paper->scissors, scissors->rock
    ai_move = moves[beats[predicted_human]]

    y_true = np.zeros((3,1))
    y_true[moves.index(human_move),0] = 1
    backward(x,a1,a2,y_true)

    # neuron connections
    connections = [
        # input->hidden (6x10)
        [[0.5 for _ in range(10)] for _ in range(6)],
        # hidden->output (10x3)
        [[0.5 for _ in range(3)] for _ in range(10)]
    ]


    return JSONResponse({"ai_move": ai_move, "neurons": neurons, "connections": connections})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)