# LFM3 on iPad

An iOS app that runs the LiquidAI LFM3-1.7B model on the GPU of an A12-family iPad. The app has a chat and a benchmark.

The app uses llama.cpp with the patch from [llamacpp-apple5-lfm3](https://github.com/0xbyt4/llamacpp-apple5-lfm3). The patch adds GPU kernels for Apple GPU family 5 (A12, A12X, A12Z) and speed-ups for LFM3. Without the patch, llama.cpp runs the matrix multiplications, the norms and the softmax on the CPU of these devices.

The author tested only the iPad Pro 11-inch (2nd generation, A12Z). The author did not test the other devices.

## Contents

| Path | Purpose |
|---|---|
| `LFM3Bench/App/` | The Swift sources of the app. |
| `LFM3Bench/Config/` | Build settings, the Info.plist keys and the entitlements file. |
| `LFM3Bench/LFM3Bench.xcodeproj` | The Xcode project. |
| `scripts/setup.sh` | Gets the patch repository, applies the patch to llama.cpp and builds `llama.xcframework`. |
| `prompts/` | The prompt files of the measured results. |
| `LICENSE` | MIT license for the files in this repository. |

## Requirements

- A Mac with Xcode 26 or later. The iOS device SDK must be installed.
- CMake, for the llama.cpp build.
- An Apple developer account, for the signing of the app.
- An iPad or iPhone with iOS 18 or later. Turn on Developer Mode on the device (Settings > Privacy & Security > Developer Mode).

The Simulator is not supported. `scripts/setup.sh` builds llama.cpp only for devices.

The app uses the "Increased Memory Limit" entitlement. The author used a paid developer account. The author did not test a free account.

## Build the app

1. Clone this repository and go into it.
2. Run `scripts/setup.sh`.

   The script clones the patch repository into `llamacpp-apple5-lfm3/` and checks out the patch commit that this app was tested with. Then it gets llama.cpp at the base commit and applies the patch. Then it builds `llamacpp-apple5-lfm3/llama.cpp/build-apple/llama.xcframework`. The Xcode project links this file. To use another URL for the patch repository, set `PATCH_REPO`. To use another patch commit, set `PATCH_REF`. If `llamacpp-apple5-lfm3/` exists, the script uses it and does not clone.

3. Copy `LFM3Bench/Config/Signing.example.xcconfig` to `LFM3Bench/Config/Signing.xcconfig`.
4. In `Signing.xcconfig`, set your team ID and a bundle identifier of your own. Xcode > Settings > Accounts shows the team ID. Git ignores this file.
5. Open `LFM3Bench/LFM3Bench.xcodeproj` in Xcode.
6. Select your device and run the `LFM3Bench` scheme.

To build from the command line:

```
xcodebuild -project LFM3Bench/LFM3Bench.xcodeproj -scheme LFM3Bench -configuration Release -destination generic/platform=iOS -allowProvisioningUpdates build
```

To check the build without signing, add `CODE_SIGNING_ALLOWED=NO` and remove `-allowProvisioningUpdates`.

## Add a model

The app loads GGUF files from its Documents folder.

1. Download the model files from [0xbyt4/LFM3-1.7B-Apple5-GGUF](https://huggingface.co/0xbyt4/LFM3-1.7B-Apple5-GGUF). To prepare the files yourself, see "Prepare a model" in the README of the patch repository.
2. Connect the device to the Mac.
3. In Finder, select the device and open the Files tab.
4. Drag the GGUF files onto "LFM3 Bench".
5. If the app runs, open the Bench tab and then the Chat tab again. The chat reads the file list when the Chat tab appears.

With Spec off, the chat selects a file with `sketch` in its name first. Then it selects a file with `embQ8` in its name. If no file matches, it selects the first file in alphabetical order.

Files with `draft` in the name show only with Spec on. With Spec on, the chat selects a file with `draftq4` in its name first, then a file with `draft` in its name.

## Use the chat

- The model menu (top right) selects a file and loads or unloads it.
- "Spec" (top left) turns on speculative decoding. The model menu then shows only the files with a draft head. The file with a draft head is `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf` on Hugging Face. Without a file with a draft head, Spec stays disabled.
- "Think" turns the thinking of the model on or off for the next message.
- "STE" asks the model to answer in ASD-STE100 Simplified Technical English. It applies to new chats.
- The sidebar shows the chats. You can delete a chat there.
- The status line shows the generation speed, the prompt speed, the used context and the thermal state. With speculative decoding, it also shows the percentage of correct draft tokens.

The context holds 8192 tokens. The cache keeps the current chat, so a new message decodes only its own tokens. When you open another chat, the app decodes all its earlier messages again. A new message needs 512 free tokens. When the context is full, the reply stops and the status line shows "Context full". Then tap New chat.

## Privacy

- The app has no network code. The model runs on the device.
- The app saves the chats as plain text in `Documents/conversations.json`. File sharing is on, so a Mac connected to the device can read this file in Finder. The device backups also include it.
- The app writes llama.cpp logs to `Documents/llama-log.txt` and benchmark results to `Documents/results.jsonl`.

## Run the benchmark

The Bench tab runs measurements on the GGUF files in Documents. Unload the chat model before you run the benchmark.

Launch arguments start a run when the app opens. Set them in Xcode (Product > Scheme > Edit Scheme > Run > Arguments) or with devicectl:

```
xcrun devicectl device process launch --device <device> <bundle-id> -- -bench all -seconds 180 -cool 1
```

| Arguments | Run |
|---|---|
| `-bench all -seconds 180 -cool 1` | Load time, prompt speed and sustained greedy generation for every file. Give a file name instead of `all` for one file. `-cool 1` (the default) waits for the device to cool down before each file. `-cool 0` turns off the wait. |
| `-bench <file> -spec 1` | The same run with speculative decoding. The file must have a draft head. |
| `-bench <file> -prompts <file>` | Takes the generation prompts from a text file in Documents. A line with only `%%` separates the prompts. The app does not apply the chat template. Write each prompt with the template tokens, as in the default prompt in `Bench.swift`. |
| `-depth <file>` | Generation speed at growing context depths. |
| `-verify <file> -at <position>` | The cost of decode batches of 1 to 8 tokens. `-at` sets the start position. The default is 1024. Use at most 6000. Add `-spec 1` to run the draft head. |
| `-attention 1` | Attention kernel timing. |
| `-kerneltest 1` | Compares the GPU kernels with the CPU. |
| `-trace <file>` | Writes the sum of every F32 tensor of one forward pass to `Documents/trace-<file>.txt`. |
| `-mmap 0` | Copies the weights into Metal buffers instead of mapping the file. Use it with one of the runs above. |

The app writes each measurement as one JSON line to `Documents/results.jsonl`.

Debug builds also accept `-remote 1`. With it, the app reads chat commands from `Documents/remote/command.json` and writes its state and screenshots to `Documents/remote/`. Release builds do not have this mode.

## Measured results

The author measured these values on 2026-10-31 on an iPad Pro 11-inch (2nd generation, A12Z) with iPadOS 27.0.1. The model file was `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf`. Each value is the mean of 3 runs of the benchmark, with greedy decoding.

| Prompt | Speculative decoding off (tok/s) | Speculative decoding on (tok/s) | Change |
|---|---|---|---|
| Essay (default) | 38.6 | 38.9 | +0.8% |
| Bash script (`prompts/bash.txt`) | 37.2 | 41.3 | +11.0% |
| Math word problem (`prompts/math.txt`) | 37.1 | 41.1 | +10.7% |
| 10 mixed prompts (`prompts/mix.txt`, 120 s) | 34.5 | 37.0 | +7.5% |

The prompt speed was 201 tok/s. The README of [llamacpp-apple5-lfm3](https://github.com/0xbyt4/llamacpp-apple5-lfm3) gives the ranges, the thermal notes and the full method.

To repeat a measurement:

1. Copy `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf` and the files in `prompts/` to the Documents folder of the app.
2. Unload the chat model.
3. Run one variant at a time. Change the order of the variants between the runs. For example:

   ```
   xcrun devicectl device process launch --device <device> --terminate-existing <bundle-id> -- -bench <model file> -seconds 60 -cool 1 -spec 0 -prompts bash.txt
   xcrun devicectl device process launch --device <device> --terminate-existing <bundle-id> -- -bench <model file> -seconds 60 -cool 1 -spec 1 -prompts bash.txt
   ```

4. Wait 60 s between the runs.
5. Read the `summary` lines in `Documents/results.jsonl`.

Without `-prompts`, the benchmark uses the essay prompt. In the chat, the author saw higher speeds than in the benchmark, but did not find the cause. This README therefore gives only the benchmark values.

## License

The files in this repository use the MIT license. llama.cpp has its own MIT license. The LFM3 model has its own license. Read it on the [model page](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).
