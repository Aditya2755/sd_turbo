# SDXL-Turbo Gradio Demo

Real-time text-to-image generation on a DigitalOcean GPU droplet using Stability AI's **SDXL-Turbo** model and a **Gradio** web UI.

---

## 1  Create a GPU Droplet

1. Log in to DigitalOcean → Create → Droplets.
2. Choose an image:
   * **Ubuntu 22.04** (recommended).
3. Select a **GPU Droplet**:
   * E.g. "`g-2vcpu-16gb` (NVIDIA L40S 24 GB VRAM)" or larger.
4. Enable SSH authentication and paste your public key.
5. (Optional) Add a firewall rule to allow port `7860` **or** use an SSH reverse-tunnel / ngrok (see below).
6. Create the droplet and note its IPv4 address.

---

## 2  Install System Dependencies (first login)

```bash
# ── Connect ───────────────────────────────────────────────
ssh root@<DROPLET_IP>

# ── System update + essentials ────────────────────────────
apt-get update && apt-get upgrade -y
apt-get install -y git-lfs python3-venv build-essential

# ── NVIDIA drivers & CUDA toolkit (already pre-installed on DO GPU images)
#     Verify:
nvidia-smi
```

---

## 3  Clone this repository

```bash
# (still on the droplet)
export REPO="sd_turbo"   # change if you forked under another name

git clone https://github.com/<YOUR_USER>/${REPO}.git
cd ${REPO}
```

If you plan to push changes back to GitHub, add the SSH remote:
```bash
git remote set-url origin git@github.com:<YOUR_USER>/${REPO}.git
```

---

## 4  Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt
```

> ℹ️ The first installation may compile `xformers`, which may take several minutes.

---

## 5  Authenticate to Hugging Face (optional but recommended)

If you have an account and possibly need to accept model terms:

```bash
echo "HF_TOKEN=hf_********************************" > .env
```
`app.py` automatically picks up `HF_TOKEN`.

---

## 6  Run the Gradio app

```bash
python app.py
```

By default Gradio listens on **`0.0.0.0:7860`**. If your droplet firewall allows the port you can open:

```
http://<DROPLET_IP>:7860
```

### Expose securely with ngrok (alternative)

```bash
# Download & install (one-time)
wget -O ngrok.tgz https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-v3-stable-linux-amd64.tgz
sudo tar -C /usr/local/bin -xzf ngrok.tgz ngrok
rm ngrok.tgz
# Add your authtoken
ngrok config add-authtoken <YOUR_NGROK_TOKEN>

# Launch the tunnel (inside a screen/tmux or add &)
ngrok http 7860 --region us &
```
`ngrok` will print a public HTTPS URL you can share.

---

## 7  Example usage

* Prompt: **"A cinematic shot of a baby raccoon wearing an intricate Italian priest robe"**
* Steps: **1** (Turbo works great with 1–4 steps)
* Guidance scale is hard-coded to `0.0` (Turbo was trained without classifier-free guidance).

Generated image appears instantly (~200 ms on an L40S).

---

## 8  Performance tips

* **xformers** is enabled automatically → reduces memory usage.
* PyTorch ≥ 2.0 🡒 the UNet is compiled via `torch.compile` once, then runs faster (~25 % speedup).
* Keep the default **512×512** resolution for best quality; bigger sizes will work but lose fidelity.
* The first run downloads ~4 GB of model weights into `~/.cache/huggingface`.

---

## 9  Updating

```bash
git pull
pip install -r requirements.txt --upgrade
```

---

## 10  Folder tree

```text
sd_turbo/
├─ app.py           ← Gradio server
├─ requirements.txt ← Python dependencies
└─ README.md        ← This guide
```

Happy hacking! 🧨 