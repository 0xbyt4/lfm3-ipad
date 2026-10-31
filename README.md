# LFM3 on iPad

An iOS app that runs the LiquidAI LFM3-1.7B model on the GPU of an A12-family iPad or iPhone. The app has a chat and a benchmark.

The app uses llama.cpp with the patch from [llamacpp-apple5-lfm3](https://github.com/0xbyt4/llamacpp-apple5-lfm3). The patch adds GPU kernels for Apple GPU family 5 (A12, A12X, A12Z) and speed-ups for LFM3. Without the patch, llama.cpp runs the matrix multiplications, the norms and the softmax on the CPU of these devices.

The author tested only the iPad Pro 11-inch (2nd generation, A12Z). The author did not test the other devices.

This is an independent project. It is not affiliated with Liquid AI.

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

- A Mac with Xcode 26 or later. The author used Xcode 26.6 and installed the app on a device with iPadOS 27.0.1.
- After you install Xcode, open it once. Accept the license and install the iOS platform. `xcode-select -p` must show the Xcode directory, for example `/Applications/Xcode.app/Contents/Developer`. If it shows the Command Line Tools, run `sudo xcode-select -s /Applications/Xcode.app`.
- CMake 3.28 or later (`brew install cmake`) and git, for the llama.cpp build.
- About 1 GB of disk space for llama.cpp and its build, and 2.3 GB for the two model files.
- An Apple account, for the signing of the app. A free account is sufficient. Add it in Xcode > Settings > Accounts.
- A device with an A12-family chip and iOS 18 or later:
  - A12 Bionic: iPhone XS, iPhone XS Max, iPhone XR, iPad mini (5th generation), iPad Air (3rd generation), iPad (8th generation).
  - A12X Bionic: iPad Pro 11-inch (1st generation), iPad Pro 12.9-inch (3rd generation).
  - A12Z Bionic: iPad Pro 11-inch (2nd generation), iPad Pro 12.9-inch (4th generation).
- Developer Mode on the device (Settings > Privacy & Security > Developer Mode). If the switch is not there, connect the device to the Mac and open Xcode once.

Newer devices can also run the app. The patch selects its compat kernels only for GPUs without SIMD-group reduction. Apple lists this feature (Metal 3) for the A14 and later. On these devices, llama.cpp uses its normal kernels.

The Simulator is not supported. `scripts/setup.sh` builds llama.cpp only for devices.

The app uses the "Increased Memory Limit" entitlement. The author used a free Apple account (Personal Team). The provisioning profile that Xcode makes for a free account includes this entitlement. The profile is valid for 7 days. After 7 days, the app does not open. Then run the app from Xcode again. A paid developer account gives a profile that is valid for 1 year.

## Build the app

1. Clone this repository and go into it.
2. Run `scripts/setup.sh`.

   The script clones the patch repository into `llamacpp-apple5-lfm3/` and checks out the patch commit that this app was tested with. Then it gets llama.cpp at the base commit and applies the patch. Then it builds `llamacpp-apple5-lfm3/llama.cpp/build-apple/llama.xcframework`. The Xcode project links this file. To use another URL for the patch repository, set `PATCH_REPO`. To use another patch commit, set `PATCH_REF`.

   If `llamacpp-apple5-lfm3/` exists, the script uses it and does not clone. The directory must be at `PATCH_REF`. Else the script stops with a message. To use the directory at any commit, set `PATCH_REF=HEAD`.

3. Copy `LFM3Bench/Config/Signing.example.xcconfig` to `LFM3Bench/Config/Signing.xcconfig`.
4. In `Signing.xcconfig`, set your team ID and a bundle identifier of your own. Git ignores this file. Do not set the team in the Signing & Capabilities editor of Xcode: the editor writes the team ID into `project.pbxproj`. To find the team ID:
   - With a paid developer account: the Membership details page of your account on developer.apple.com shows it.
   - With a free account: in Xcode > Settings > Accounts, select your account, click Manage Certificates and add an "Apple Development" certificate if there is none. Then run this command. The team ID is the `OU` value:

     ```
     security find-certificate -c "Apple Development" -p | openssl x509 -noout -subject
     ```

5. Connect the device to the Mac with a cable. On the device, tap Trust when it asks if you trust this computer.
6. Open `LFM3Bench/LFM3Bench.xcodeproj` in Xcode.
7. Select your device and run the `LFM3Bench` scheme.
8. With a free account, iOS does not open the app the first time and shows "Untrusted Developer". On the device, open Settings > General > VPN & Device Management, select your Apple account and tap Trust. Then open the app.

To build and install from the command line:

1. Build:

   ```
   xcodebuild -project LFM3Bench/LFM3Bench.xcodeproj -scheme LFM3Bench -configuration Release -destination generic/platform=iOS -derivedDataPath LFM3Bench/build -allowProvisioningUpdates build
   ```

2. Find the identifier of your device: `xcrun devicectl list devices`.
3. Install:

   ```
   xcrun devicectl device install app --device <device> LFM3Bench/build/Build/Products/Release-iphoneos/LFM3Bench.app
   ```

If the build stops with a provisioning error, connect the device and run the app from Xcode once. To check the build without signing, add `CODE_SIGNING_ALLOWED=NO` to the build command and remove `-allowProvisioningUpdates`.

### Update

1. Run `git pull`.
2. Delete `llamacpp-apple5-lfm3/`.
3. Run `scripts/setup.sh`.
4. Build and run the app again.

## Add a model

The app loads GGUF files from its Documents folder.

The model files are on Hugging Face, in [0xbyt4/LFM3-1.7B-Apple5-GGUF](https://huggingface.co/0xbyt4/LFM3-1.7B-Apple5-GGUF):

- `LFM3-1.7B-QAD-Q4_0-embQ8-sketch.gguf` (1.15 GB): the chat without speculative decoding.
- `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf` (1.16 GB): the chat with speculative decoding (Spec). The benchmark results below used this file.

One file is sufficient. The draftq4 file also runs without Spec. To prepare the files yourself, see "Prepare a model" in the README of the patch repository.

To add a model with a Mac:

1. Download the file. On the model page, open "Files and versions" and click the download icon of the file. Or use the Hugging Face CLI (`brew install hf` or `pip install -U huggingface_hub`):

   ```
   hf download 0xbyt4/LFM3-1.7B-Apple5-GGUF LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf --local-dir .
   ```

2. Connect the device to the Mac.
3. In Finder, select the device and open the Files tab.
4. Drag the GGUF file onto "LFM3 Bench".
5. If the app is open, tap Bench, then tap Chat. The chat reads the file list each time the Chat tab opens.

To add a model without a Mac:

1. On the device, open the model page in Safari and download the file from "Files and versions". Safari saves it in the Downloads folder.
2. In the Files app, move the file to On My iPad > LFM3 Bench (On My iPhone on an iPhone).
3. If the app is open, tap Bench, then tap Chat.

With Spec off, the model menu shows all GGUF files. The chat selects a file without `draft` in its name before a file with `draft` in its name. In each of these two groups, it selects a file with `sketch` in its name first, then a file with `embQ8` in its name, then the first file in the sorted list (upper case before lower case).

With Spec on, the model menu shows only the files with `draft` in the name. The chat selects a file with `draftq4` in its name first, then a file with `draft` in its name.

When the app lists the GGUF files (when the Chat or Bench tab opens), it excludes them from the device backups.

## Use the chat

- The model menu (top right) selects a file and loads or unloads it.
- "Spec" (top left) turns on speculative decoding. The model menu then shows only the files with `draft` in the name. The app finds a draft head only by this name. The file with a draft head is `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf` on Hugging Face. Without a file with a draft head, Spec stays disabled. When you turn Spec on or off, a loaded model loads again.
- "Think" turns the thinking of the model on or off for the next message.
- "STE" asks the model to answer in ASD-STE100 Simplified Technical English. It applies to new chats.
- The sidebar shows the chats. You can delete a chat there.
- The status line shows the generation speed, the prompt speed, the used context and the thermal state. With speculative decoding, it also shows the percentage of correct draft tokens. When a reply ends early, an orange note shows the cause, for example "Context full" or "Stopped".

The context holds 8192 tokens. The cache keeps the current chat, so a new message decodes only its own tokens. When you open another chat, the app decodes all its earlier messages again. A new message needs free space for its own tokens and 512 more tokens. When the context is full, the reply stops and the status line shows "Context full". Then start a new chat: tap the pencil icon (top left), or New chat in the sidebar.

The app stops the generation when it goes to the background. Keep the app in the foreground while it answers.

## Privacy

- The app has no network code. The model runs on the device.
- The app saves the chats as plain text in `Documents/conversations.json`. File sharing is on, so a Mac connected to the device can read this file in Finder. The Files app of the device also shows the Documents folder. The device backups include this file.
- The model can write links in its answers. A tap on a link opens it in Safari.
- The app writes llama.cpp logs to `Documents/llama-log.txt` and benchmark results to `Documents/results.jsonl`.

## Run the benchmark

The Bench tab runs measurements on the GGUF files in Documents. Unload the chat model before you run the benchmark. Keep the app in the foreground: the app stops a run when it goes to the background.

Launch arguments start a run when the app opens. Set them in Xcode (Product > Scheme > Edit Scheme > Run > Arguments) or with devicectl:

```
xcrun devicectl device process launch --device <device> <bundle-id> -- -bench all -seconds 180 -cool 1
```

`xcrun devicectl list devices` shows the identifier of the device. `<bundle-id>` is the bundle identifier in `Signing.xcconfig`.

| Arguments | Run |
|---|---|
| `-bench all -seconds 180 -cool 1` | Load time, prompt speed and sustained greedy generation for every file. Give a file name instead of `all` for one file. `-cool 1` (the default) waits for the nominal thermal state before each file, for at most 10 minutes. `-cool 0` turns off the wait. `-cool` also applies to `-depth` and `-verify`. |
| `-bench <file> -spec 1` | The same run with speculative decoding. The file must have a draft head. |
| `-bench <file> -prompts <file>` | Takes the generation prompts from a text file in Documents. A line with only `%%` separates the prompts. The app does not apply the chat template. Write each prompt with the template tokens, as in the default prompt in `Bench.swift`. If the file does not exist, the run does not start. |
| `-depth <file>` | Generation speed at growing context depths. |
| `-verify <file> -at <position>` | The cost of decode batches of 1 to 8 tokens. `-at` sets the start position, from 1 to 6000. The default is 1024. Add `-spec 1` to run the draft head. |
| `-attention 1` | Attention kernel timing. |
| `-kerneltest 1` | Compares the GPU kernels with the CPU. |
| `-trace <file>` | Writes the sum of every F32 tensor of one forward pass to `Documents/trace-<file>.txt`. |
| `-mmap 0` | Copies the weights into Metal buffers instead of mapping the file. Use it with one of the runs above. |

Give one run type at a time. If you give more than one, the app runs only the first in this order: `-kerneltest`, `-verify`, `-attention`, `-trace`, `-depth`, `-bench`.

The app writes each measurement as one JSON line to `Documents/results.jsonl`. The `start` and `summary` lines of `-bench` name the prompts (`essay` or the file name) and tell if speculative decoding was on.

Debug builds also accept `-remote 1`. With it, the app reads chat commands from `Documents/remote/command.json` and writes its state and screenshots to `Documents/remote/`. Release builds do not have this mode.

## Measured results

The author measured these values on 2026-10-31 on an iPad Pro 11-inch (2nd generation, A12Z) with iPadOS 27.0.1. The model file was `LFM3-1.7B-QAD-Q4_0-embQ8-sketch-draftq4.gguf`. Each value is the mean of 3 runs of the benchmark, with greedy decoding. The change is computed from the unrounded means.

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

## Troubleshooting

- **`Signing for "LFM3Bench" requires a development team`.** `LFM3Bench/Config/Signing.xcconfig` does not exist. Do steps 3 and 4 of "Build the app".
- **`No Accounts` or `No profiles for '<bundle identifier>' were found`.** Xcode has no account for the team in `Signing.xcconfig`. Add your account in Xcode > Settings > Accounts, and check the team ID.
- **`You have not agreed to the Xcode license agreements`.** Run `sudo xcodebuild -license accept`, or open Xcode and accept the license.
- **The device is not in the device list of Xcode.** Connect the device with a cable and tap Trust on the device. Turn on Developer Mode (see "Requirements").
- **"Untrusted Developer" when the app opens.** Do step 8 of "Build the app".
- **The app opened before, but now it does not open.** With a free account, the provisioning profile is valid for 7 days. Run the app from Xcode again. The models and the chats stay.
- **The model menu is empty.** Documents has no GGUF file. See "Add a model".
- **"Load failed, see llama-log.txt".** Open the Files tab of the device in Finder and copy `llama-log.txt` to the Mac. The log shows the cause.
- **`setup.sh` stops with "... is not at ...".** The patch repository in `llamacpp-apple5-lfm3/` is at another commit. Do the steps in "Update".

## License

The files in this repository use the MIT license. llama.cpp has its own MIT license. The LFM3 model has its own license. Read it on the [model page](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF).
