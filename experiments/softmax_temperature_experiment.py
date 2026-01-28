import numpy as np
import matplotlib.pyplot as plt


def softmax(x, temperature=1.0):
    x = x / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def main():
    x = np.linspace(-5, 5, 100)

    temperatures = [0.3, 0.7, 1.0, 2.0]

    plt.figure()
    for t in temperatures:
        y = softmax(x, temperature=t)
        plt.plot(x, y, label=f"T={t}")

    plt.title("Effect of temperature on softmax distribution")
    plt.xlabel("Input value")
    plt.ylabel("Softmax probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/softmax_temperature.png", dpi=200)
    plt.close()

    print("Saved figures/softmax_temperature.png")


if __name__ == "__main__":
    main()
