import matplotlib.pyplot as plt


def plot_channel_signal(df, channel):
    focused_df = df[df["state"] == "focused"]
    unfocused_df = df[df["state"] == "unfocused"]
    drownsy_df = df[df["state"] == "drownsy"]

    # plt.figure(figsize=(16, 6))
    plt.plot(focused_df["t"], focused_df[channel])
    plt.plot(unfocused_df["t"], unfocused_df[channel])
    plt.plot(drownsy_df["t"], drownsy_df[channel])
    plt.title(f"{channel} signal by time")
    plt.xlabel("Time")
    plt.ylabel(f"{channel} Signal")
    plt.legend(["Focused", "Unfocused", "Drownsy"])
    plt.tight_layout()
    # plt.show()
