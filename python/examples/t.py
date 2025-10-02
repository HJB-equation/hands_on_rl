import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

freq = st.slider("选择频率", 1, 10, 5)

x = np.linspace(0, 10, 400)
y = np.sin(freq * x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title(f"sin({freq}x)")

st.pyplot(fig)
