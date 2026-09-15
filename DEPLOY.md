# Deploying WatchNext to Oracle Cloud (Always Free)

The bot is one always-on Python process that long-polls Telegram. It makes
only **outbound** connections (Telegram, Groq, TMDB), so you never open an
inbound port. Only **one** copy may run at a time — stop the bot on your
laptop before starting it on the server, or Telegram will error on both.

Two runtime facts specific to this project:

- **The model files are gitignored.** `git clone` gives you code but not
  `data/movies.db`, `data/model.joblib`, `data/embeddings.npz`,
  `data/collab.npz`. You copy those (~50 MB) to the server. You do **not**
  need the 1.6 GB `data/movielens/` — that is only for rebuilding.
- **`.env` is gitignored too** (it holds your tokens). You copy it up
  separately; never commit it.

---

## 1. Create the free VM

1. Sign in at <https://cloud.oracle.com> → **Compute → Instances → Create**.
2. **Image:** Canonical Ubuntu 24.04.
3. **Shape:** Change shape → **Ampere (Arm)** → `VM.Standard.A1.Flex`.
   Set **1 OCPU / 6 GB RAM** (well within the Always Free allowance and
   more than enough). Avoid the tiny AMD `E2.1.Micro` — 1 GB RAM is tight
   once scikit-learn loads.
4. **SSH keys:** upload your public key (or let Oracle generate one and
   download the private key).
5. Create it, and note the **public IP**.

You do **not** need to touch the security list / open any ports.

SSH in (default user is `ubuntu`):

```bash
ssh ubuntu@<PUBLIC_IP>
```

---

## 2. Get the code and data onto the VM

On the **server**, install git and clone the code:

```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/naveensenthuran100-ux/WatchNext.git
```

> If PR #1 isn't merged yet, check out that branch instead:
> `cd WatchNext && git fetch origin rebuild-accuracy && git checkout rebuild-accuracy`

From your **laptop** (in the project folder), copy the model files and
`.env` up — this is the part `git` can't do:

```bash
scp .env ubuntu@<PUBLIC_IP>:~/WatchNext/.env
scp data/movies.db data/model.joblib data/embeddings.npz data/collab.npz \
    ubuntu@<PUBLIC_IP>:~/WatchNext/data/
```

(If `~/WatchNext/data/` doesn't exist yet, run `mkdir -p ~/WatchNext/data`
on the server first.)

---

## 3. Run it — pick ONE of the two paths

### Path A — systemd (recommended: fewest moving parts)

On the server:

```bash
sudo apt install -y python3-venv
cd ~/WatchNext
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

Sanity-check it starts (Ctrl-C after you see `Application started`):

```bash
./venv/bin/python -m src.bot
```

Install the service so it runs forever and survives reboots:

```bash
sudo cp deploy/watchnext.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now watchnext
```

Manage it:

```bash
systemctl status watchnext        # is it running?
journalctl -u watchnext -f        # live logs
sudo systemctl restart watchnext  # after pulling new code
sudo systemctl stop watchnext     # stop it
```

To update later: `cd ~/WatchNext && git pull && sudo systemctl restart watchnext`.

### Path B — Docker (most reproducible)

On the server:

```bash
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu    # then log out and back in once
cd ~/WatchNext
docker build -t watchnext .
docker run -d --name watchnext --restart unless-stopped \
    --env-file .env \
    -v ~/WatchNext/data:/app/data \
    watchnext
```

The `-v` mount keeps the SQLite DB (your votes, cached profiles) and model
files on the host, so they survive rebuilds. Manage it:

```bash
docker logs -f watchnext          # live logs
docker restart watchnext
docker stop watchnext
```

To update later: `git pull && docker build -t watchnext . && docker restart watchnext`
(rebuild picks up new code).

---

## 4. Confirm it's live

Message your bot on Telegram: `/status` should report the film count, and
`/rec something like Interstellar` should reply. First free-text query
downloads the ~30 MB embedding model from Hugging Face (one-off, cached).

## Notes

- **RAM:** expect ~0.5–1 GB resident once loaded. The 6 GB shape has ample
  headroom.
- **Rebuilding on the server** (only if you expand the corpus) needs the
  MovieLens data and TMDB calls: `python -m src.train && python -m src.embed
  && python -m src.collab`. Day to day you don't touch this.
- **Costs:** the Ampere A1 Always Free shape has no time limit. Just don't
  exceed the free allowance (4 OCPU / 24 GB total across instances).
