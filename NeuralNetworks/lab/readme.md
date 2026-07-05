# Keras vs. PyTorch – oppsummering fra Iris-oppgaven

## Overordnet filosofi

| | Keras/TensorFlow | PyTorch |
|---|---|---|
| Stil | Høynivå, deklarativt | Lavnivå, eksplisitt |
| Trening | Innebygd `.fit()` | Manuell training loop |
| Læringskurve | Raskere å komme i gang | Mer kode, men bedre innsikt i hva som skjer |
| Mest brukt i | Produksjon/deployment | Forskning/akademia |

## 1. Definere modellen

**Keras** – bygges som en liste av lag:
```python
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])
```

**PyTorch** – defineres som en klasse med egen `forward()`-metode:
```python
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisNet()
```

## 2. Loss-funksjon og label-format

| | Keras | PyTorch |
|---|---|---|
| Loss ved one-hot labels | `categorical_crossentropy` | Ikke støttet direkte |
| Loss ved heltall-labels | `sparse_categorical_crossentropy` | `nn.CrossEntropyLoss()` |
| Softmax i modellen | Eksplisitt i siste lag | **Ikke** i `forward()` – `CrossEntropyLoss` regner softmax internt |

Viktig lærdom: PyTorch sin `CrossEntropyLoss` krever heltall-labels (0,1,2), ikke one-hot.

## 3. Kompilering vs. manuell oppsett

**Keras** – samler optimizer, loss og metrikker i ett kall:
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**PyTorch** – defineres hver for seg, ingen `.compile()`:
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
```

## 4. Trening

**Keras** – ett kall, validation innebygd via parameter:
```python
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

**PyTorch** – manuell loop, du styrer hvert steg selv:
```python
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
```

## 5. Validation

- **Keras**: `validation_split=0.2` tar automatisk 20% av *treningsdataen* og bruker det til validering underveis – ingen manuell splitting nødvendig.
- **PyTorch**: må settes opp manuelt (egen `train_test_split` eller egen loop-logikk) – ingen innebygd støtte.

## 6. Evaluering

**Keras** – ett kall gjør jobben:
```python
loss_all, accuracy_all = model.evaluate(features_norm, labels_onehot)
```

**PyTorch** – manuell forward pass med `no_grad()`:
```python
model.eval()
with torch.no_grad():
    outputs = model(X_all_tensor)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_all_tensor).float().mean()
```

## 7. Praktisk lærdom fra oppgaven

- Learning rate hadde stor effekt: `lr=0.001` (Adam default) var for lavt for Iris innen 100 epochs, `lr=0.1` ga rask konvergens.
- Testresultat: 100% accuracy på testsett (30 samples), 98.67% på hele datasettet (150 samples) – viser at lite testsett kan gi et for optimistisk bilde.
- One-hot vs. heltall-labels må velges bevisst tidlig, siden det påvirker både loss-funksjon og evt. konverteringsbehov senere.