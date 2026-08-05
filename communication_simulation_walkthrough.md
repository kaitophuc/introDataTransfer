# Communication Simulation Walkthrough

This note records the step-by-step path we followed from raw bits to a simple OFDM simulation. The goal was not to jump straight to library magic, but to understand what each transmitter, channel, receiver, and evaluation step means.

## 1. Starting Point: Bits

We started with a short bit sequence:

```text
1 0 1 1 0 0 1 0
```

These bits are abstract data. To send them through a channel, the transmitter must convert them into signal values.

## 2. BPSK Modulation

We chose BPSK because it is the simplest digital modulation:

```text
0 -> -1
1 -> +1
```

Example:

```text
bits:    1  0  1  1  0  0  1  0
symbols: +1 -1 +1 +1 -1 -1 +1 -1
```

The receiver decision rule is:

```text
received > 0 -> bit 1
received < 0 -> bit 0
```

## 3. AWGN Channel

The first channel model was noise only:

```text
y = x + n
```

Where:

```text
x = transmitted BPSK symbol
n = noise
y = received value
```

We first used hand-picked noise values to see how a symbol can be pushed across zero and decoded incorrectly.

## 4. Bit Errors And BER

After decoding, we compared:

```text
original bits
decoded bits
```

Then counted bit errors:

```text
bit_errors = number of positions where decoded != original
```

The bit error rate is:

```text
BER = bit_errors / total_bits
```

Example:

```text
2 wrong bits out of 8 -> BER = 2 / 8 = 0.25
```

## 5. Random Noise With Torch

We moved from hand-picked noise to random Gaussian noise:

```python
noise = noise_std * torch.randn(size=(num_bits,), device=device)
```

Then:

```python
received = symbols + noise
decoded_bits = (received > 0).long()
bit_errors = (decoded_bits != bits).sum()
ber = bit_errors / num_bits
```

We used `num_bits = 1000`, then increased it to `100000` to make BER more stable.

## 6. Noise Power

Noise power means average squared noise strength.

For real noise:

```text
noise_power = mean(n^2)
```

For complex noise:

```text
noise_power = mean(|n|^2)
```

If:

```text
noise_std = 0.5
```

then:

```text
noise_power = noise_std^2 = 0.25
```

## 7. SNR

SNR means signal-to-noise ratio:

```text
SNR = signal_power / noise_power
```

In dB:

```text
SNR_dB = 10 * log10(SNR)
```

For BPSK symbols `-1` and `+1`, transmit signal power is:

```text
1
```

We changed from sweeping `noise_std` to sweeping:

```python
snr_dbs = [0, 2, 4, 6, 8, 10, 12, 14]
```

Then computed:

```python
snr_linear_target = 10 ** (snr_db_target / 10)
noise_power_target = 1 / snr_linear_target
```

## 8. Real Fading Channel

We introduced a real channel coefficient:

```text
y = h*x + n
```

Examples:

```text
h = 0.5  -> signal is weakened
h = -0.5 -> signal is weakened and sign-flipped
h = 0.1  -> deep fade
```

If the receiver knows `h`, it can equalize:

```python
equalized = received / h
decoded_bits = (equalized > 0).long()
```

Substituting the channel equation:

```text
equalized = (h*x + n) / h
equalized = x + n/h
```

This shows why small `h` is dangerous: it amplifies noise.

## 9. Random Real Fading

We then used a different real `h` for every bit:

```python
h = torch.randn(size=(num_bits,), device=device)
```

This created random fading. Some values of `h` were close to zero, which caused deep fades.

We separated errors into:

```text
deep fade:    |h| < 0.2
good channel: |h| >= 0.2
```

Deep-fade BER was much worse than good-channel BER.

## 10. Complex Fading

Real wireless baseband channels are complex:

```text
h = h_real + j*h_imag
```

Manual normalized complex channel:

```python
h = (h_real + 1j * h_imag) / sqrt(2)
```

We divide by `sqrt(2)`, not `2`, because power is amplitude squared:

```text
divide amplitude by sqrt(2) -> divide power by 2
```

For complex fading, power is:

```python
torch.mean(torch.abs(h) ** 2)
```

The receiver equalizes:

```python
equalized = received / h
```

For BPSK, decisions use the real part:

```python
decoded_bits = (torch.real(equalized) > 0).long()
```

## 11. Sionna Version

We then rebuilt the same simulation using Sionna blocks.

Manual Torch pieces became:

```text
torch.randint      -> BinarySource
manual complex h   -> complex_normal
manual AWGN noise  -> AWGN
manual error count -> count_errors
```

Imports:

```python
import torch
from sionna.phy import config
from sionna.phy.channel import AWGN
from sionna.phy.mapping import BinarySource
from sionna.phy.utils import complex_normal, count_errors
```

Seed/device:

```python
config.seed = 0
device = config.device
```

Bits:

```python
source = BinarySource(precision="single", device=device)
bits = source([num_bits]).to(torch.long)
```

Complex channel:

```python
h = complex_normal(
    (num_bits,),
    precision="single",
    device=device,
)
```

Sionna `complex_normal` is already normalized to average complex power near `1`, so we do not divide by `sqrt(2)` again.

AWGN:

```python
awgn = AWGN(precision="single", device=device)
received = awgn(clean_received, noise_power_target_tensor)
```

The noise power tensor is not empty. It wraps a Python number, such as `0.1`, into a Torch tensor on the correct device:

```python
noise_power_target_tensor = torch.tensor(
    noise_power_target,
    dtype=torch.float32,
    device=device,
)
```

## 12. Frames

We reshaped one long bit stream into frames:

```text
bits shape: [num_frames, bits_per_frame]
```

For example:

```text
frame 0: [b0 b1 b2 b3 b4 b5 b6 b7]
frame 1: [b8 b9 b10 b11 b12 b13 b14 b15]
```

Code:

```python
bits_per_frame = 8
num_frames = num_bits // bits_per_frame
bits = source([num_frames, bits_per_frame]).to(torch.long)
```

## 13. FER

BER counts individual bit errors.

FER counts frames with at least one bit error:

```python
errors = decoded_bits != bits
frame_errors = torch.any(errors, dim=1)
num_frame_errors = frame_errors.sum()
fer = num_frame_errors / num_frames
```

FER is usually larger than BER because each frame has multiple chances to contain an error.

## 14. OFDM-Like Shape

We renamed the frame dimension to OFDM language:

```python
num_subcarriers = 8
num_frames = num_bits // num_subcarriers
bits = source([num_frames, num_subcarriers]).to(torch.long)
```

Now:

```text
each row    = one OFDM symbol
each column = one subcarrier
```

BPSK frequency-domain subcarrier symbols:

```python
x_freq = (bits * 2 - 1).to(torch.float32)
```

## 15. IFFT And FFT

OFDM has two views:

```text
frequency domain: subcarrier symbols
time domain: transmitted waveform samples
```

Transmitter:

```python
x_time = torch.fft.ifft(x_freq.to(torch.complex64), dim=1, norm="ortho")
```

Receiver check:

```python
x_freq_recovered = torch.fft.fft(x_time, dim=1, norm="ortho")
recovery_error = torch.max(torch.abs(x_freq_recovered - x_freq))
```

We verified:

```text
FFT(IFFT(x_freq)) ~= x_freq
```

with only tiny floating-point error.

## 16. AWGN-Only OFDM Chain

We tested:

```text
x_freq -> IFFT -> x_time
x_time + AWGN -> y_time
FFT -> y_freq
decode BPSK
```

Code:

```python
y_time = awgn(x_time, noise_power_target_tensor)
y_freq = torch.fft.fft(y_time, dim=1, norm="ortho")
decoded_bits = (torch.real(y_freq) > 0).long()
```

At high SNR, BER became very low.

## 17. Flat Fading In OFDM

We added one complex channel value per frame:

```python
h_frame = complex_normal(
    (num_frames, 1),
    precision="single",
    device=device,
)
```

Shape:

```text
h_frame: [num_frames, 1]
x_time:  [num_frames, num_subcarriers]
```

PyTorch broadcasts one channel value across all samples/subcarriers in a frame.

Time-domain flat fading:

```python
clean_y_time = h_frame * x_time
y_time = awgn(clean_y_time, noise_power_target_tensor)
y_freq = torch.fft.fft(y_time, dim=1, norm="ortho")
equalized_freq = y_freq / h_frame
decoded_bits = (torch.real(equalized_freq) > 0).long()
```

Flat fading means all subcarriers in a frame share the same channel coefficient.

## 18. Frequency-Selective Fading

Frequency-selective means different frequencies experience different channel effects.

In OFDM:

```text
each subcarrier = one frequency
```

So:

```text
flat fading:             one h for all subcarriers
frequency-selective:     one H per subcarrier
```

Code:

```python
h_freq = complex_normal(
    (num_frames, num_subcarriers),
    precision="single",
    device=device,
)
```

Simplified frequency-domain OFDM model:

```python
clean_y_freq = h_freq * x_freq.to(torch.complex64)
y_freq = awgn(clean_y_freq, noise_power_target_tensor)
equalized_freq = y_freq / h_freq
decoded_bits = (torch.real(equalized_freq) > 0).long()
```

This is:

```text
Y_freq[k] = H_freq[k] * X_freq[k] + N_freq[k]
```

Important correction we made:

```python
clean_y_freq = h_freq * x_time
```

was wrong because it mixed frequency-domain channel values with time-domain samples.

Correct:

```python
clean_y_freq = h_freq * x_freq.to(torch.complex64)
```

## 19. Current `main_OFDM.py` Meaning

The current code is a simplified OFDM receiver-side model:

```text
bits
-> BPSK subcarrier symbols X_freq
-> random frequency-domain channel H_freq
-> AWGN
-> equalize by H_freq
-> decode
-> BER and FER
```

Although `x_time = IFFT(x_freq)` exists, the current frequency-selective channel path uses the frequency-domain model directly. That is valid for a simplified OFDM simulation.

## 20. Cyclic Prefix Concept

A real time-domain multipath channel creates echoes. That means each received sample can depend on multiple transmitted samples.

This is convolution.

Problem:

```text
the start of one OFDM symbol can be polluted by the end of the previous symbol
```

This is inter-symbol interference.

Cyclic prefix fixes this by copying the end of the OFDM time-domain symbol and placing it at the front.

Example:

```text
x_time = [x0 x1 x2 x3 x4 x5 x6 x7]
cp_len = 2
cyclic prefix = [x6 x7]
transmitted = [x6 x7 x0 x1 x2 x3 x4 x5 x6 x7]
```

If the cyclic prefix is long enough for the channel echoes, then after removing the prefix and taking FFT, the messy time-domain convolution becomes simple multiplication per subcarrier:

```text
Y_freq[k] = H_freq[k] X_freq[k] + N_freq[k]
```

## 21. Cyclic Prefix Code Step

The next code step was to add cyclic prefix only, without multipath yet:

```python
cp_len = 2

cyclic_prefix = x_time[:, -cp_len:]
x_time_cp = torch.cat([cyclic_prefix, x_time], dim=1)
```

Expected shapes:

```text
x_time shape:         [num_frames, 8]
cyclic_prefix shape:  [num_frames, 2]
x_time_cp shape:      [num_frames, 10]
```

Meaning:

```text
x_time_cp = cyclic-prefix-extended OFDM time-domain signal
```

## 22. Time-Domain Multipath Channel

We then replaced the random frequency-domain shortcut with a simple physical time-domain channel:

```python
h_time = torch.tensor(
    [1.0 + 0.0j, 0.5 + 0.3j],
    dtype=torch.complex64,
    device=device,
)
```

This is a 2-tap channel:

```text
h_time[0] = direct path
h_time[1] = delayed echo path
```

For this channel:

```text
y[n] = h_time[0] * x[n] + h_time[1] * x[n-1]
```

The first term is the current sample through the direct path. The second term is the previous sample arriving late as an echo.

For an 8-subcarrier OFDM symbol, the 2-tap channel is treated as:

```text
[h0 h1 0 0 0 0 0 0]
```

The FFT gives one channel value per subcarrier:

```python
h_freq_from_time = torch.fft.fft(
    h_time,
    n=num_subcarriers,
    dim=0,
)

h_freq = h_freq_from_time.view(1, num_subcarriers)
```

Shape meaning:

```text
h_time shape:           [2]
h_freq_from_time shape: [8]
h_freq shape:           [1, 8]
```

The `[1, 8]` shape lets PyTorch broadcast the same 8-subcarrier channel across all OFDM frames.

## 23. OFDM With Cyclic Prefix And `h_time`

The current simulation chain is:

```text
bits
-> BPSK symbols on subcarriers
-> IFFT to time domain
-> add cyclic prefix
-> pass through h_time channel
-> add AWGN noise in time domain
-> remove cyclic prefix
-> FFT back to frequency domain
-> equalize using h_freq
-> decode bits
-> compute BER and FER
```

The transmitter creates one BPSK symbol per subcarrier:

```python
x_freq = (bits * 2 - 1).to(torch.float32)
```

IFFT converts subcarrier symbols into a time-domain OFDM waveform:

```python
x_time = torch.fft.ifft(x_freq.to(torch.complex64), dim=1, norm="ortho")
```

The cyclic prefix copies the last `cp_len` samples and puts them in front:

```python
cyclic_prefix = x_time[:, -cp_len:]
x_time_cp = torch.cat([cyclic_prefix, x_time], dim=1)
```

The channel loop performs time-domain convolution:

```python
y_time_cp_clean = torch.zeros_like(x_time_cp)

for tap_index, tap_value in enumerate(h_time):
    y_time_cp_clean[:, tap_index:] += tap_value * x_time_cp[:, :x_time_cp.shape[1] - tap_index]
```

For the 2-tap channel, this means:

```text
y[n] = h0*x[n] + h1*x[n-1]
```

Noise is added in the time domain:

```python
y_time_cp = awgn(y_time_cp_clean, noise_power_target_tensor)
```

The receiver removes the cyclic prefix:

```python
y_time_no_cp = y_time_cp[:, cp_len:cp_len + num_subcarriers]
```

Then FFT converts the received time-domain OFDM symbol back to subcarriers:

```python
y_freq = torch.fft.fft(y_time_no_cp, dim=1, norm="ortho")
```

Because the cyclic prefix is long enough, the time-domain convolution becomes simple multiplication in frequency:

```text
Y_freq[k] = H_freq[k] * X_freq[k] + N_freq[k]
```

So the receiver equalizes by dividing each subcarrier by its channel:

```python
equalized_freq = y_freq / h_freq
decoded_bits = (torch.real(equalized_freq) > 0).long()
```

## 24. BER Per Subcarrier

We added BER per subcarrier:

```python
errors = decoded_bits != bits
ber_per_subcarrier = errors.float().mean(dim=0)
```

This showed an important OFDM idea:

```text
weak subcarriers have more bit errors
```

The channel magnitudes were:

```text
[1.53, 1.57, 1.39, 1.03, 0.58, 0.46, 0.86, 1.27]
```

Subcarrier index `5` had the weakest channel magnitude, about `0.46`, and it also had the highest BER.

This happens because equalization divides by the channel:

```text
equalized = received / H[k]
```

If `|H[k]|` is small, then the noise on that subcarrier gets amplified more.

## 25. 16-QAM Modulation

We replaced BPSK with 16-QAM.

BPSK maps:

```text
1 bit -> 1 real symbol
```

16-QAM maps:

```text
4 bits -> 1 complex symbol
```

We used Gray mapping on each axis:

```text
00 -> -3
01 -> -1
11 -> +1
10 -> +3
```

The first two bits choose the real part, and the last two bits choose the imaginary part:

```text
[b0 b1 b2 b3]
-> real from b0,b1
-> imaginary from b2,b3
```

The raw 16-QAM constellation has average symbol power `10`, so we normalized by `sqrt(10)`:

```python
x_freq = x_freq / sqrt(10)
```

After normalization, average symbol power was close to `1`.

16-QAM had worse BER than BPSK at the same SNR because its constellation points are closer together.

## 26. Cleaner Code Blocks

We cleaned the code by naming the main communication blocks:

```python
qam16_modulate()
qam16_demodulate()
ofdm_ifft()
add_cyclic_prefix()
apply_time_domain_channel()
remove_cyclic_prefix()
ofdm_fft()
equalize_frequency_domain()
```

This made the main program read like the communication block diagram:

```text
bits
-> 16-QAM mapper
-> OFDM IFFT
-> cyclic prefix
-> time-domain channel
-> AWGN
-> remove cyclic prefix
-> OFDM FFT
-> equalization
-> 16-QAM demapper
-> BER/FER
```

## 27. Resource Grid

We changed from one OFDM symbol per frame to multiple OFDM symbols per frame.

Old shape:

```text
bits shape:   [num_frames, num_subcarriers, 4]
x_freq shape: [num_frames, num_subcarriers]
```

New shape:

```text
bits shape:   [num_frames, num_data_ofdm_symbols, num_subcarriers, 4]
x_grid shape: [num_frames, num_ofdm_symbols, num_subcarriers]
```

We chose:

```python
num_frames = 100000
num_subcarriers = 8
num_ofdm_symbols = 4
num_pilot_symbols = 1
num_data_ofdm_symbols = 3
bits_per_qam_symbol = 4
```

So:

```text
bits shape: [100000, 3, 8, 4]
x_grid shape: [100000, 4, 8]
```

Each frame now contains:

```text
OFDM symbol 0: pilot
OFDM symbol 1: data
OFDM symbol 2: data
OFDM symbol 3: data
```

Each data frame carries:

```text
3 data OFDM symbols * 8 subcarriers * 4 bits = 96 bits
```

## 28. OFDM Functions On The Last Dimension

With the resource grid, the last dimension is the subcarrier/sample dimension:

```text
[frames, ofdm_symbols, subcarriers]
```

So FFT/IFFT use:

```python
dim=-1
```

The cyclic prefix also works on the last dimension:

```python
cyclic_prefix = x_time[..., -cp_len:]
x_time_cp = torch.cat([cyclic_prefix, x_time], dim=-1)
```

Removing CP also slices the last dimension:

```python
y_time_no_cp = y_time_cp[..., cp_len:cp_len + num_subcarriers]
```

The time-domain channel loop was also updated to use the last dimension:

```python
for tap_index, tap_value in enumerate(h_time):
    y_time_cp_clean[..., tap_index:] += tap_value * x_time_cp[..., :x_time_cp.shape[-1] - tap_index]
```

We checked that:

```text
time-domain channel + CP + FFT
```

matches:

```text
h_freq * x_grid
```

with only tiny floating-point error.

## 29. Pilot-Based Channel Estimation

The receiver should not magically know the channel.

Instead, it uses pilots:

```text
known transmitted pilot + received pilot -> channel estimate
```

The frequency-domain pilot equation is:

```text
Y_pilot[k] = H[k] * X_pilot[k] + N[k]
```

So the receiver estimates:

```text
H_hat[k] = Y_pilot[k] / X_pilot[k]
```

In our grid:

```python
pilot_y_freq = y_grid[:, :num_pilot_symbols, :]
data_y_freq = y_grid[:, num_pilot_symbols:, :]
```

Shapes:

```text
pilot_y_freq shape: [100000, 1, 8]
data_y_freq shape:  [100000, 3, 8]
```

The channel estimate is:

```python
h_hat_per_frame = pilot_y_freq / pilot_x_freq
h_hat = torch.mean(h_hat_per_frame, dim=1, keepdim=True)
```

Since we currently have one pilot OFDM symbol, the mean does not change the estimate, but it keeps the code ready for multiple pilot symbols.

The estimate has shape:

```text
h_hat shape: [100000, 1, 8]
```

That means each frame has one estimated channel value per subcarrier.

## 30. Estimated Channel Vs Perfect Channel

Perfect-channel equalization uses the true channel:

```python
perfect_equalized_data_freq = data_y_freq / h_freq
```

Estimated-channel equalization uses the pilot estimate:

```python
equalized_data_freq = data_y_freq / h_hat
```

The estimated-channel receiver is more realistic, but worse:

```text
estimated channel BER > perfect channel BER
```

We also measured channel estimation MSE:

```python
channel_estimation_mse = torch.mean(torch.abs(h_hat - h_freq) ** 2)
```

As SNR increases:

```text
channel estimation MSE decreases
estimated-channel BER gets closer to perfect-channel BER
```

This connects channel estimation quality directly to data decoding quality.

## Current Learning Position

We now have a small but realistic OFDM resource-grid simulation:

```text
16-QAM
resource grid
pilot OFDM symbol
data OFDM symbols
cyclic prefix
time-domain multipath channel
AWGN
pilot-based channel estimation
frequency-domain equalization
BER/FER measurement
```

## 31. Sionna Mapper And Demapper

We replaced our manual 16-QAM mapper and demapper with Sionna:

```python
mapper = Mapper(
    constellation_type="qam",
    num_bits_per_symbol=bits_per_qam_symbol,
    precision="single",
    device=device,
)
```

The mapper converts groups of bits into QAM symbols:

```text
bits shape: [100000, 3, 8, 4]
mapped symbols before squeeze: [100000, 3, 8, 1]
mapped symbols after squeeze:  [100000, 3, 8]
```

The final `1` means four bits became one 16-QAM symbol.

We also changed the demapper to produce soft values:

```python
demapper = Demapper(
    demapping_method="app",
    constellation_type="qam",
    num_bits_per_symbol=bits_per_qam_symbol,
    hard_out=False,
    precision="single",
    device=device,
)
```

With `hard_out=False`, the demapper returns LLRs:

```text
LLR > 0 means bit is probably 1
LLR < 0 means bit is probably 0
LLR near 0 means unsure
large |LLR| means confident
```

We convert LLRs back to hard bits with:

```python
decoded_bits = (llr > 0).to(torch.long)
```

This keeps BER/FER measurement simple while preserving a useful soft output for later coding or ML.

## 32. Sionna ResourceGrid

A resource grid is the time-frequency table for OFDM.

For our current setup:

```text
4 OFDM symbols
8 subcarriers
1 pilot OFDM symbol
3 data OFDM symbols
```

The grid is:

```text
OFDM symbol 0: P P P P P P P P
OFDM symbol 1: D D D D D D D D
OFDM symbol 2: D D D D D D D D
OFDM symbol 3: D D D D D D D D
```

`P` means pilot. `D` means data.

We created it with:

```python
rg = ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=num_subcarriers,
    subcarrier_spacing=15e3,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=cp_len,
    dc_null=False,
    pilot_pattern="kronecker",
    pilot_ofdm_symbol_indices=[0],
    precision="single",
    device=device,
)
```

Sionna computed:

```text
num pilot symbols: 8
num data symbols: 24
```

because:

```text
total resource elements = 4 * 8 = 32
pilot resource elements = 1 * 8 = 8
data resource elements = 32 - 8 = 24
```

The `kronecker` pilot pattern means Sionna automatically creates a structured pilot pattern. With one transmitter and one stream, it simply means the selected OFDM symbol carries pilots across the subcarriers.

## 33. Sionna ResourceGridMapper

The `ResourceGrid` defines the empty map.

The `ResourceGridMapper` fills the map.

It takes flattened data QAM symbols:

```text
[100000, 1, 1, 24]
```

and places them into the data positions of the grid:

```python
x_freq_sionna_input = x_freq.reshape(num_frames, 1, 1, rg.num_data_symbols)
x_grid_sionna = rg_mapper(x_freq_sionna_input)
```

The output shape is:

```text
x_grid_sionna shape: [100000, 1, 1, 4, 8]
```

Meaning:

```text
100000 frames
1 transmitter
1 stream
4 OFDM symbols
8 subcarriers
```

Compared with our earlier manual grid:

```text
manual grid: [100000, 4, 8]
Sionna grid: [100000, 1, 1, 4, 8]
```

Sionna adds transmitter and stream dimensions.

## 34. Sionna OFDM Modulator And Demodulator

We moved from manual OFDM helpers to Sionna OFDM blocks.

The modulator replaces:

```text
IFFT + cyclic prefix insertion
```

The demodulator replaces:

```text
cyclic prefix removal + FFT
```

Current blocks:

```python
ofdm_modulator = OFDMModulator(
    cyclic_prefix_length=cp_len,
    precision="single",
    device=device,
)

ofdm_demodulator = OFDMDemodulator(
    fft_size=num_subcarriers,
    l_min=0,
    cyclic_prefix_length=cp_len,
    precision="single",
    device=device,
)
```

The clean no-channel check gave:

```text
Sionna OFDM recovery error: 5.960464477539062e-07
```

That means:

```text
x_grid_sionna
-> OFDMModulator
-> OFDMDemodulator
-> recovered x_grid_sionna
```

works up to tiny floating-point error.

Sionna serializes time samples:

```text
x_grid_sionna shape: [100000, 1, 1, 4, 8]
x_time_sionna shape: [100000, 1, 1, 40]
```

The `40` comes from:

```text
4 OFDM symbols * (8 samples + 2 CP samples) = 40
```

## 35. Frequency Order: ifftshift And fftshift

Sionna and raw Torch FFT use different frequency-bin ordering.

Sionna stores subcarriers like:

```text
[-4, -3, -2, -1, 0, +1, +2, +3]
```

Raw `torch.fft.ifft()` expects:

```text
[0, +1, +2, +3, -4, -3, -2, -1]
```

So Sionna's OFDM modulator does:

```python
torch.fft.ifftshift(...)
```

before the IFFT.

On the receiver side, after FFT, Sionna does:

```python
torch.fft.fftshift(...)
```

to return to Sionna's centered frequency order.

We verified this:

```text
old manual OFDM error: 6.380920886993408
Sionna-style manual OFDM error: 0.0
```

The large old error did not mean the math was broken. It only meant the subcarrier ordering convention was different.

Because our channel frequency response is compared with `h_hat`, we also shifted the true channel:

```python
h_freq_from_time = torch.fft.fft(h_time, n=num_subcarriers, dim=0)
h_freq_from_time = torch.fft.fftshift(h_freq_from_time, dim=0)
```

Now `h_freq` and `h_hat` use the same Sionna frequency order.

## 36. Sionna TimeChannel And Rayleigh Block Fading

We replaced the manual channel function with Sionna's channel model:

```python
channel_model = RayleighBlockFading(
    num_rx=1,
    num_rx_ant=1,
    num_tx=1,
    num_tx_ant=1,
    precision="single",
    device=device,
)

sionna_time_channel = TimeChannel(
    channel_model=channel_model,
    bandwidth=15e3 * num_subcarriers,
    num_time_samples=num_ofdm_symbols * (num_subcarriers + cp_len),
    l_min=0,
    l_max=0,
    normalize_channel=True,
    return_channel=True,
    precision="single",
    device=device,
)
```

The transmitted time-domain waveform is:

```python
x_time_sionna = ofdm_modulator(x_grid_sionna)
```

Shape:

```text
x_time_sionna shape: [100000, 1, 1, 40]
```

Then the Sionna time channel gives:

```python
y_time_clean, h_time_sionna = sionna_time_channel(x_time_sionna)
```

`y_time_clean` means the received waveform after the wireless channel but before AWGN:

```text
x_time_sionna -> channel -> y_time_clean
```

`h_time_sionna` is the actual channel that Sionna generated.

For our current channel:

```text
Rayleigh block fading
l_min = 0
l_max = 0
```

That means:

```text
one random complex channel value per frame
same channel across the whole frame
one tap only
flat fading across subcarriers
```

The printed channel shape was:

```text
h_time_sionna shape: [100000, 1, 1, 1, 1, 40, 1]
```

Meaning:

```text
100000 frames
1 receiver
1 receive antenna
1 transmitter
1 transmit antenna
40 time samples
1 tap
```

An indexing example:

```python
true_h = h_time_sionna[:, 0, 0, 0, 0, 0, 0]
```

This means:

```text
all frames
receiver 0
receive antenna 0
transmitter 0
transmit antenna 0
time sample 0
tap 0
```

The result has shape:

```text
[100000]
```

So it gives one true channel value per frame.

Because block fading is constant inside the frame, taking time sample `0` is enough.

## 37. Sionna Channel Estimation MSE

With the Sionna Rayleigh channel, the true channel is no longer a fixed manual `h_freq`.

Instead, the true channel comes from:

```python
true_h = h_time_sionna[:, 0, 0, 0, 0, 0, 0]
true_h_freq = true_h.view(num_frames, 1, 1)
```

The estimated channel is still from the pilot:

```python
h_hat_per_frame = pilot_y_freq / pilot_x_freq
h_hat = torch.mean(h_hat_per_frame, dim=1, keepdim=True)
```

Then the estimation error is:

```python
channel_estimation_mse = torch.mean(torch.abs(h_hat - true_h_freq) ** 2)
```

The result decreased as SNR increased:

```text
SNR 0 dB  -> MSE about 0.997
SNR 8 dB  -> MSE about 0.158
SNR 16 dB -> MSE about 0.025
SNR 24 dB -> MSE about 0.00398
SNR 28 dB -> MSE about 0.00158
```

This makes sense because pilot channel estimation is:

```text
Y_pilot = H * X_pilot + noise
H_hat = Y_pilot / X_pilot
H_hat = H + noise / X_pilot
```

Less noise means `H_hat` is closer to the true channel.

## 38. Current Clean Sionna/Torch Split

The current active pipeline is:

```text
bits
-> Sionna Mapper
-> Sionna ResourceGridMapper
-> Sionna OFDMModulator
-> Sionna TimeChannel with RayleighBlockFading
-> Sionna AWGN
-> Sionna OFDMDemodulator
-> Sionna LSChannelEstimator
-> Sionna LMMSEEqualizer
-> Sionna Demapper
-> decoded bits
-> BER/FER
```

The main manual pieces left are:

```text
reshaping tensors between learning-friendly shapes and Sionna shapes
BER/FER calculation
```

The receiver now estimates the channel from pilots and equalizes the data symbols without using the true hidden channel from the simulator.

## 39. Road Ahead

From here, the big path is:

```text
working OFDM link
-> channel coding
-> soft decoding
-> realistic multipath channel
-> pilot patterns and interpolation
-> end-to-end Sionna link
-> ML / AI receiver
-> GPU and custom kernel optimization
```

The next major steps are:

1. Channel coding

Add redundancy so the receiver can fix some errors. Start with a simple repetition code, then later move to Sionna LDPC.

2. Soft decoding

Use LLR confidence values instead of only hard `0` and `1` decisions. This is why `Demapper(hard_out=False)` matters.

3. More realistic channel

Move from flat Rayleigh block fading to multipath or frequency-selective fading. Then different subcarriers can experience different channel strengths.

4. Pilot patterns and interpolation

Use pilots that are scattered across time and frequency instead of one full pilot OFDM symbol. The receiver must estimate missing channel values between pilot positions.

5. End-to-end Sionna link

Use more standard Sionna blocks for the full chain:

```text
encoder
-> mapper
-> resource grid
-> OFDM
-> channel
-> channel estimator
-> equalizer
-> demapper
-> decoder
```

6. ML / AI receiver

Train a neural receiver, neural demapper, or neural channel estimator. Compare it against the classical `LSChannelEstimator + LMMSEEqualizer` receiver.

7. GPU and custom kernel optimization

After the communication algorithm is clear, compare the Sionna/Torch implementation with custom CUDA kernels such as the `kaito_kernel_*` files.

The important learning direction is:

```text
understand the communication system first
-> replace pieces with Sionna blocks
-> replace selected receiver pieces with ML
-> optimize performance later
```


## 40. Repetition Coding And Soft Decoding

We first added a simple repetition code to understand channel coding.

The uncoded frame carried:

```text
96 useful bits per frame
```

With repetition-3 coding, the frame still transmitted 96 coded bits, but only carried:

```text
32 useful information bits per frame
```

The code rate was:

```text
R = information bits / coded bits = 32 / 96 = 1/3
```

Hard repetition decoding used majority vote:

```text
decoded repeated bits: 1, 0, 1
majority -> 1
```

Soft repetition decoding used LLRs:

```text
LLRs: +0.5, +0.4, -3.0
sum = -2.1
decision -> 0
```

This showed an important receiver idea:

```text
hard decisions throw away confidence
soft decisions keep confidence
```

Soft decoding performed better because the receiver used how confident each repeated copy was, not just whether each copy looked like `0` or `1`.

## 41. LDPC Channel Coding

We then replaced the toy repetition code with a real Sionna LDPC code.

LDPC means:

```text
Low-Density Parity-Check code
```

It protects information bits by adding structured redundancy through many sparse parity-check rules.

The LDPC chain is:

```text
info_bits
-> LDPC5GEncoder
-> coded_bits
-> QAM mapper
-> OFDM transmitter
-> channel
-> OFDM receiver
-> demapper LLRs
-> LDPC5GDecoder
-> decoded_info_bits
```

For the LDPC-sized setup, we used:

```text
num_frames = 1000
num_subcarriers = 64
num_ofdm_symbols = 14
bits_per_qam_symbol = 4
code_rate = 1/2
```

With one pilot OFDM symbol and 13 data OFDM symbols:

```text
13 data OFDM symbols * 64 subcarriers * 4 bits = 3328 coded bits/frame
3328 * 1/2 = 1664 information bits/frame
```

Important shapes:

```text
info_bits:          [1000, 1664]
coded_bits_flat:    [1000, 3328]
coded_bits:         [1000, 13, 64, 4]
x_freq:             [1000, 13, 64]
x_grid_sionna:      [1000, 1, 1, 14, 64]
x_time_sionna:      [1000, 1, 1, 924]     with CP length 2
llr_flat:           [1000, 3328]
decoded_info_bits:  [1000, 1664]
```

The LDPC decoder consumes LLRs directly:

```text
demapper LLRs
-> LDPC decoder belief updates
-> final decoded bits
```

That means LDPC is a soft-input decoder. It uses confidence values, not only hard `0/1` decisions.

## 42. TDL Multipath Channel

We replaced flat Rayleigh fading with a TDL channel:

```python
channel_model = TDL(
    model="A",
    delay_spread=300e-9,
    carrier_frequency=3.5e9,
    min_speed=0.0,
    max_speed=0.0,
    num_rx_ant=1,
    num_tx_ant=1,
    precision="single",
    device=device,
)
```

TDL means:

```text
Tapped Delay Line
```

It models multiple delayed copies of the transmitted signal:

```text
y[t] = h0*x[t] + h1*x[t-1] + h2*x[t-2] + ... + noise
```

For our TDL-A setup, Sionna used 16 time-domain channel taps.

That changed the channel from:

```text
flat fading:
one channel value for all subcarriers
```

to:

```text
frequency-selective fading:
different subcarriers can see different channel gains
```

Because the channel had 16 taps, we increased cyclic prefix length:

```text
cp_len = 16
```

With 14 OFDM symbols, 64 subcarriers, and CP length 16:

```text
x_time length = 14 * (64 + 16) = 1120
```

The channel output length became:

```text
y_time length = 1120 + (16 - 1) = 1135
```

So the observed shapes were:

```text
x_time_sionna: [1000, 1, 1, 1120]
y_time_clean:  [1000, 1, 1, 1135]
h_time_sionna: [1000, 1, 1, 1, 1, 1135, 16]
```

The last dimension of `h_time_sionna` is the number of channel taps.

## 43. Channel Timing And `l_min`

For the TDL channel, Sionna used channel tap indices:

```text
l_min = -6
l_max = 9
```

That gives:

```text
9 - (-6) + 1 = 16 taps
```

`l_min` is the earliest channel tap index. It is a timing reference used by the OFDM demodulator for phase compensation.

So the demodulator should use the same timing reference as the channel:

```python
OFDMDemodulator(
    fft_size=num_subcarriers,
    l_min=sionna_time_channel.l_min,
    cyclic_prefix_length=cp_len,
    precision="single",
    device=device,
)
```

The practical meaning is:

```text
channel and receiver must agree on where the OFDM symbol timing starts
```

## 44. Mobility And Doppler

We then made the TDL channel time-varying by setting:

```python
min_speed=10.0
max_speed=10.0
```

This means:

```text
10 m/s, about 22 mph
```

Now the channel is:

```text
multipath
frequency-selective
time-varying
```

With nearest-neighbor interpolation:

```python
interpolation_type="nn"
```

the BER stopped improving at high SNR. This is an error floor.

The cause was no longer mainly noise. The cause was channel-estimation and interpolation error:

```text
the channel changes over time
nearest-neighbor copying is too crude
```

Changing the estimator to linear interpolation:

```python
interpolation_type="lin"
```

fixed the error floor in our run.

Important clarification:

```text
"nn" in Sionna interpolation means nearest neighbor
not neural network
```

## 45. Pilot Pattern Tradeoff

A pilot is a known signal sent so the receiver can estimate the channel.

The receiver knows:

```text
pilot_x = transmitted pilot
pilot_y = received pilot
```

Then it estimates:

```text
h_hat = pilot_y / pilot_x
```

With 14 OFDM symbols and 64 subcarriers, one full pilot OFDM symbol means:

```text
64 pilot resource elements
```

Two pilot OFDM symbols means:

```text
128 pilot resource elements
```

Three pilot OFDM symbols means:

```text
192 pilot resource elements
```

We compared pilot layouts under mobility:

```text
2 pilots: [2, 11]
3 pilots: [0, 7, 13]
```

The 3-pilot layout improved reliability:

```text
at 8 dB:
2 pilots BER ~ 0.1349
3 pilots BER ~ 0.1006

at 12 dB:
2 pilots BER ~ 0.000584
3 pilots BER ~ 0.000175
```

But it reduced data capacity:

```text
2 pilots -> 12 data OFDM symbols/frame
3 pilots -> 11 data OFDM symbols/frame
```

This is a core wireless tradeoff:

```text
more pilots
-> better channel estimation
-> better BER/FER
-> fewer resources for data
```

Current strong classical baseline:

```text
LDPC rate 1/2
16-QAM
64 subcarriers
14 OFDM symbols
TDL-A multipath channel
10 m/s mobility
CP length 16
3 full pilot OFDM symbols at [0, 7, 13]
LS channel estimation
linear interpolation
LMMSE equalization
soft demapping
LDPC decoding
```

## 46. Full-Grid Neural Receiver Plan

The current neural receiver is really a neural demapper:

```text
received grid
-> LS channel estimation
-> LMMSE equalization
-> neural demapper
-> LDPC decoder
```

It only sees one equalized data resource element at a time:

```text
[real(equalized symbol), imag(equalized symbol), log(no_eff)]
```

The full-grid neural receiver goal is bigger:

```text
received OFDM resource grid
-> neural receiver
-> coded-bit LLRs
-> LDPC decoder
```

That means the neural network learns to use pilots, neighboring subcarriers, neighboring OFDM symbols, and noise information. It starts to replace the classical receiver blocks:

```text
LS channel estimation
LMMSE equalization
classical demapping
```

The transmitter, channel, OFDM modulator/demodulator, and LDPC decoder stay the same at first. This keeps the experiment controlled.

### Step 47. Freeze The Baseline

Purpose:

```text
keep a trusted classical result to compare against
```

Baseline path:

```text
BinarySource
-> LDPC encoder
-> 16-QAM mapper
-> ResourceGridMapper
-> OFDMModulator
-> TDL-A TimeChannel
-> AWGN
-> OFDMDemodulator
-> LSChannelEstimator
-> LMMSEEqualizer
-> Sionna Demapper
-> LDPC decoder
```

Done when:

```text
classical BER/FER still runs and chart generation still works
```

### Step 48. Expose The Right Training Tensors

Purpose:

```text
the full-grid neural receiver needs the received grid before classical channel estimation/equalization
```

Training input should come from:

```text
y_grid_sionna
```

Target labels should come from:

```text
coded_bits
```

Important shapes:

```text
y_grid_sionna: [batch, 1, 1, 14, 64]
coded_bits:    [batch, rg.num_data_symbols, 4]
```

Done when:

```text
we can print y_grid_sionna shape and coded_bits shape from one batch
```

### Step 49. Build Full-Grid Features

Purpose:

```text
convert the complex received grid into real neural-network input channels
```

First simple feature stack:

```text
channel 0: real(y_grid_sionna)
channel 1: imag(y_grid_sionna)
channel 2: pilot mask
channel 3: data mask
channel 4: log(noise_power)
```

Target feature shape:

```text
grid_features: [batch, 5, 14, 64]
```

Done when:

```text
grid_features shape is correct and all values are finite
```

### Step 50. Decide Output Convention

Purpose:

```text
the neural receiver should output soft bits for data resource elements only
```

Target output shape:

```text
predicted_llr: [batch, rg.num_data_symbols, 4]
```

Then flatten for LDPC:

```text
llr_flat: [batch, num_coded_bits_per_frame]
```

Done when:

```text
predicted_llr and coded_bits have matching shape
```

### Step 51. Create A Small FullGridNeuralReceiver

Purpose:

```text
start with a simple CNN over the 2D OFDM grid
```

First model idea:

```text
Conv2d input channels 5
-> ReLU
-> Conv2d hidden channels
-> ReLU
-> Conv2d output channels 4
```

The model processes the whole grid:

```text
[batch, 5, 14, 64]
-> [batch, 4, 14, 64]
```

Then we gather only data resource elements:

```text
[batch, 4, 14, 64]
-> [batch, rg.num_data_symbols, 4]
```

Done when:

```text
one forward pass runs without training and output shape is correct
```

### Step 52. Train Without LDPC First

Purpose:

```text
teach the neural receiver to output correct coded-bit logits
```

Loss:

```python
BCEWithLogitsLoss(predicted_llr, coded_bits.float())
```

This is the same idea as the current neural demapper training, but now the network sees the whole resource grid.

Done when:

```text
training loss decreases
coded-bit accuracy increases
```

### Step 53. Evaluate With LDPC Decoder

Purpose:

```text
measure the real communication result, not only coded-bit accuracy
```

Evaluation path:

```text
y_grid_sionna
-> FullGridNeuralReceiver
-> llr_flat
-> LDPC decoder
-> decoded_info_bits
-> BER/FER
```

Compare against:

```text
classical LS + LMMSE + Sionna Demapper
current neural demapper after LMMSE
```

Done when:

```text
we have BER/FER curves for classical, neural demapper, and full-grid neural receiver
```

### Step 54. Make The Comparison Fair

Purpose:

```text
avoid comparing receivers on different random channels and different random bits
```

Use the same generated batch for:

```text
classical receiver
neural demapper
full-grid neural receiver
```

Done when:

```text
one batch can produce all receiver outputs from the same y_grid_sionna
```

### Step 55. Train Across SNRs

Purpose:

```text
make the receiver useful across a range of channel qualities
```

Instead of training only at 12 dB, sample SNR randomly:

```text
training SNR range: 0 dB to 20 dB
```

Done when:

```text
the full-grid receiver does not only work near one SNR
```

### Step 56. Improve The Neural Architecture

Purpose:

```text
move from a tiny CNN toward a DeepRx-style full-grid receiver
```

Possible upgrades:

```text
residual CNN blocks
batch normalization or layer normalization
more hidden channels
separate pilot/data feature channels
noise-power conditioning
2D attention later if CNN is not enough
```

Done when:

```text
the full-grid receiver beats or approaches the classical receiver in at least part of the SNR range
```

### Step 57. Pilot And Channel Experiments

Purpose:

```text
test whether the neural receiver can use pilots better than the classical receiver
```

Experiments:

```text
reduce pilot density
move pilot positions
compare linear interpolation vs full-grid neural receiver
test TDL-A, TDL-B, TDL-C
test multiple speeds
```

Done when:

```text
we know where the neural receiver gives a real advantage
```

### Step 58. Save, Load, And Reuse Models

Purpose:

```text
avoid retraining from zero every run
```

Add:

```text
save model weights
load model weights
separate train mode and evaluate mode
record config with each result
```

Done when:

```text
we can train once, save, reload, and reproduce BER/FER
```

### Step 59. Research-Level Extensions

After the full-grid receiver works, the next frontier directions are:

```text
SIMO/MIMO
MU-MIMO
dynamic MCS
learned or reduced pilots
site-specific channel training
hardware impairments such as CFO, phase noise, and low-resolution ADC
real-time inference constraints
```

These are later steps. The first serious milestone is:

```text
replace LS + LMMSE + demapper with one full-grid neural receiver
while keeping the same transmitter, channel, and LDPC decoder
```

