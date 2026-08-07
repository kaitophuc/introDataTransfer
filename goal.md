# Research Goal And Roadmap

This file is the main direction for the next steps.

## Current System

The current simulation path is:

```text
Sionna BinarySource
-> LDPC encoder
-> 16-QAM mapper
-> Sionna ResourceGrid with scattered pilots
-> OFDM modulator
-> TDL-A channel + AWGN
-> OFDM demodulator
```

Then the receiver has two paths:

```text
Neural path:
full received OFDM grid
-> CNN full-grid neural receiver
-> LDPC decoder
-> BER/FER
```

```text
Classical path:
received OFDM grid
-> LS channel estimation
-> LMMSE equalizer
-> APP demapper
-> LDPC decoder
-> BER/FER
```

## Main Goal

The main research goal is to improve the current full-grid neural receiver until the simulation becomes closer to frontier neural receiver research.

We already moved beyond a neural demapper only. The neural receiver now sees the received OFDM grid and outputs coded-bit LLRs for the LDPC decoder.

## Big Steps To Catch Up Toward Frontier Research

1. Make the current full-grid neural receiver scientifically solid.
   Run larger evaluations, multiple random seeds, confidence intervals, saved CSV results, clean train/test separation, and reliable plots.

2. Improve the neural receiver input.
   Add richer full-grid features such as pilot value channels, normalized time/frequency coordinate channels, estimated channel information, and error variance.

3. Improve the neural architecture.
   Move beyond a small CNN toward deeper CNNs, residual CNNs, U-Net-style models, transformers, or attention over the time-frequency resource grid.

4. Train across more realistic variation.
   Train across many SNRs, delay spreads, Doppler speeds, TDL/CDL channel models, pilot layouts, and eventually modulation/code-rate settings.

   Delay spread stress test:
   The current mobility-trained residual neural receiver was evaluated at delay_spread=1000e-9 without retraining.
   It still outperformed the classical LS/LMMSE receiver from 8-16 dB, although BER/FER degraded compared with the 300 ns baseline.
   This suggests the neural receiver has some robustness to stronger frequency selectivity, but future training should randomize delay_spread.

5. Move from SISO to MIMO.
   Current system is one transmit antenna and one receive antenna. Frontier systems often use MIMO, MU-MIMO, or massive MIMO.

6. Support adaptive MCS.
   Current system uses fixed 16-QAM and fixed LDPC rate 1/2. Real systems use QPSK, 16-QAM, 64-QAM, 256-QAM, and multiple code rates.

7. Study pilot efficiency.
   Test reduced pilots, better pilot placement, learned pilot design, superimposed pilots, or data-aided channel estimation.

8. Add real-world impairments.
   Add carrier frequency offset, phase noise, timing offset, imperfect synchronization, power-amplifier distortion, quantization, or low-resolution ADC effects.

9. Make the simulation closer to 5G NR.
   Move toward realistic numerology, slots, DMRS-like pilots, transport blocks, MCS tables, CDL channels, and PUSCH-like structure.

10. Optimize deployment.
    Study inference speed, GPU memory, batching, model size, latency, ONNX/TensorRT export, and real-time feasibility.

## Immediate Next Direction

The next technical improvement should be:

```text
Improve the full-grid neural receiver input.
```

Start with:

```text
received grid real part
received grid imaginary part
noise feature
pilot mask
data mask
pilot value real part
pilot value imaginary part
normalized OFDM-symbol index
normalized subcarrier index
```

This keeps the current system structure stable while giving the neural receiver more useful information.
