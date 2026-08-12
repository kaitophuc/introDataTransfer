DEFAULT_WORKFLOW_NOTE = "\n".join(
    [
        "Workflow: Sionna BinarySource -> 5G LDPC encoder -> 16-QAM mapper -> 2-stream 2x2 MIMO ResourceGrid.",
        "Pilot/channel: interleaved pilots in OFDM symbols 2 and 11 -> OFDM modulator -> TDL-A time channel -> AWGN -> OFDM demodulator.",
        "Neural features: received grid + log noise power + pilot/data masks + complex pilot values + time/frequency coordinates + LS channel estimates/error variances.",
        "Neural path: six-block full-grid residual CNN -> coded-bit LLRs for both spatial streams -> LDPC decoder.",
        "Classical path: received grid + noise power -> LS channel estimator (linear interpolation) -> LMMSE equalizer -> APP demapper -> LDPC decoder.",
    ]
)


def plot_receiver_comparison(
    snr_dbs,
    neural_bers,
    classical_bers,
    neural_fers,
    classical_fers,
    output_path="results/neural_vs_classical_receiver.png",
    workflow_note=None,
):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_floor = 1e-8

    neural_bers_plot = _floor_for_log_plot(neural_bers, plot_floor)
    classical_bers_plot = _floor_for_log_plot(classical_bers, plot_floor)
    neural_fers_plot = _floor_for_log_plot(neural_fers, plot_floor)
    classical_fers_plot = _floor_for_log_plot(classical_fers, plot_floor)

    if workflow_note is None:
        workflow_note = DEFAULT_WORKFLOW_NOTE

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    axes[0].semilogy(
        snr_dbs,
        neural_bers_plot,
        marker="o",
        label="Neural receiver",
    )
    axes[0].semilogy(
        snr_dbs,
        classical_bers_plot,
        marker="s",
        label="Classical receiver",
    )
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("BER")
    axes[0].set_title("Bit Error Rate")
    axes[0].grid(True, which="both", linestyle=":")
    axes[0].legend()

    axes[1].semilogy(
        snr_dbs,
        neural_fers_plot,
        marker="o",
        label="Neural receiver",
    )
    axes[1].semilogy(
        snr_dbs,
        classical_fers_plot,
        marker="s",
        label="Classical receiver",
    )
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("FER")
    axes[1].set_title("Frame Error Rate")
    axes[1].grid(True, which="both", linestyle=":")
    axes[1].legend()

    fig.text(
        0.5,
        0.03,
        workflow_note,
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#f7f7f7",
            "edgecolor": "#cfcfcf",
        },
    )

    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_receiver_comparison_from_csv(
    csv_path="results/neural_vs_classical_receiver.csv",
    output_path="results/neural_vs_classical_receiver.png",
    workflow_note=None,
):
    import csv

    snr_dbs = []
    neural_bers = []
    classical_bers = []
    neural_fers = []
    classical_fers = []

    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            snr_dbs.append(float(row["snr_db"]))
            neural_bers.append(float(row["neural_ber"]))
            classical_bers.append(float(row["classical_ber"]))
            neural_fers.append(float(row["neural_fer"]))
            classical_fers.append(float(row["classical_fer"]))

    return plot_receiver_comparison(
        snr_dbs,
        neural_bers,
        classical_bers,
        neural_fers,
        classical_fers,
        output_path=output_path,
        workflow_note=workflow_note,
    )


def _floor_for_log_plot(values, plot_floor):
    return [max(float(value), plot_floor) for value in values]


if __name__ == "__main__":
    plot_receiver_comparison_from_csv()
