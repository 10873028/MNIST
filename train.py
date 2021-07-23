import streamlit as st
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from model import MNISTModel
from sklearn import metrics

st.title("MNIST training dashboard")

device = "cpu"
if st.checkbox("Use GPU"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
st.write("Using Device: ", device)

fullData = torchvision.datasets.MNIST(
    root="./MNIST",
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    download=False,
)

percent = st.slider(
    "How many percentage for the train data?", min_value=0.5, max_value=0.9, step=0.01
)

N = len(fullData)
trainL = int(percent * N)
testL = N - trainL

trainData, testData = Data.random_split(fullData, [trainL, testL])
st.write("Train Data Size: ", trainL)
st.write("Test Data Size: ", testL)

batch_size = st.slider("How many batch size for dataloader?", min_value=1, max_value=512, step=2)

trainLoader = Data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = Data.DataLoader(dataset=testData, batch_size=testL, shuffle=True)

model = MNISTModel(inChannels=1, outChannels=32).to(device)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = st.slider("How may epochs for training?", min_value=0, max_value=100, step=1)
if st.button("Start Trainging"):
    for _ in range(epochs):
        for trainX, trainY in trainLoader:
            trainX = trainX.to(device)
            trainY = trainY.to(device)

            result = model(trainX)
            loss = criterion(result, trainY)
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(loss.item())

    with torch.no_grad():
        model.eval()
        testX, testY = next(iter(testLoader))
        testX, testY = testX.to(device), testY.numpy()
        pred = model(testX)
        _, result = torch.max(pred, 1)
        result = result.cpu().numpy()
        st.write("accuracy: ", metrics.accuracy_score(testY, result))
        st.image(
            testX[0].squeeze(0).cpu().numpy(),
            width=280,
            clamp=True,
            caption=f"prediction: {result[0]}",
        )
