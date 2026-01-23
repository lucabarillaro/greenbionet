class TegraStatsSampler:
	"""
	Background reader for `tegrastats`.

	interval_ms: sampling interval (default 200 ms)
	rail:
	  - "auto" (default): prefer VDD_GPU_SOC + VDD_CPU_CV sum, else POM_5V_IN, else POM_5V_GPU, else GPU mW.
	  - "orin_sum": VDD_GPU_SOC + VDD_CPU_CV
	  - "VDD_GPU_SOC", "VDD_CPU_CV", "POM_5V_IN", "POM_5V_GPU", "GPU" for explicit rails
	save_trace_path: optional JSONL storing raw parsed samples.
	"""
	def __init__(self, interval_ms=200, rail="auto", save_trace_path=None):
		self.interval_ms = int(interval_ms)
		self.rail = str(rail)
		self.save_trace_path = save_trace_path
		self.proc = None
		self._thr = None
		self._stop = threading.Event()
		self._samples = []
		self._lock = threading.Lock()

		self.re_vdd_gpu_soc = re.compile(r"\bVDD_GPU_SOC\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_vdd_cpu_cv  = re.compile(r"\bVDD_CPU_CV\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_5v_in       = re.compile(r"\bPOM_5V_IN\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_5v_gpu      = re.compile(r"\bPOM_5V_GPU\s+(\d+(?:\.\d+)?)(m?W)\b", re.I)
		self.re_gpu_fallback= re.compile(r"\bGPU\b[^\n]*?(\d+(?:\.\d+)?)(m?W)\b", re.I)

	@staticmethod
	def _to_watts(val_str, unit):
		v = float(val_str); unit = (unit or "").lower()
		return v/1000.0 if unit == "mw" else v

	def __enter__(self):
		tegra = shutil.which("tegrastats") or "/usr/bin/tegrastats"
		if not Path(tegra).exists():
			print("[tegrastats] not found at", tegra)
			return self
		cmd = [tegra, "--interval", str(self.interval_ms)]
		self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
									 text=True, bufsize=1, universal_newlines=True)
		self._thr = threading.Thread(target=self._reader, daemon=True); self._thr.start()
		time.sleep(max(0.001, self.interval_ms/1000.0))  # allow first line
		return self

	def _parse_line(self, line: str):
		out = {"VDD_GPU_SOC": None, "VDD_CPU_CV": None, "POM_5V_IN": None, "POM_5V_GPU": None, "GPU": None}
		m = self.re_vdd_gpu_soc.search(line);  out["VDD_GPU_SOC"] = self._to_watts(*m.groups()) if m else None
		m = self.re_vdd_cpu_cv.search(line);   out["VDD_CPU_CV"]  = self._to_watts(*m.groups()) if m else None
		m = self.re_5v_in.search(line);        out["POM_5V_IN"]   = self._to_watts(*m.groups()) if m else None
		m = self.re_5v_gpu.search(line);       out["POM_5V_GPU"]  = self._to_watts(*m.groups()) if m else None
		m = self.re_gpu_fallback.search(line); out["GPU"]         = self._to_watts(*m.groups()) if m else None
		return out

	def _pick_power(self, rails: dict):
		mode = self.rail.lower()
		if mode in ("orin_sum","auto"):
			g, c = rails.get("VDD_GPU_SOC"), rails.get("VDD_CPU_CV")
			if (g is not None) or (c is not None):
				return (g or 0.0) + (c or 0.0)
			if mode != "auto": return None
		if mode in ("pom_5v_in","auto"):
			v = rails.get("POM_5V_IN")
			if v is not None: return v
			if mode != "auto": return None
		if mode in ("pom_5v_gpu","auto"):
			v = rails.get("POM_5V_GPU")
			if v is not None: return v
			if mode != "auto": return None
		if mode == "vdd_gpu_soc":
			return rails.get("VDD_GPU_SOC")
		if mode == "vdd_cpu_cv":
			return rails.get("VDD_CPU_CV")
		if mode in ("gpu","auto"):
			v = rails.get("GPU")
			if v is not None: return v
		return None

	def _reader(self):
		try:
			while not self._stop.is_set():
				if self.proc is None: break
				line = self.proc.stdout.readline()
				if not line:
					if self.proc.poll() is not None: break
					time.sleep(0.01); continue
				rails = self._parse_line(line)
				p = self._pick_power(rails)
				if p is not None:
					with self._lock: self._samples.append((time.time(), p))
					if self.save_trace_path:
						with open(self.save_trace_path, "a") as f:
							f.write(json.dumps({"ts": time.time(), "power_w": p, "rails": rails, "raw": line.strip()})+"\n")
		except Exception as e:
			print("[tegrastats] reader error:", e)

	def __exit__(self, *exc):
		time.sleep(max(0.001, self.interval_ms/1000.0))
		self._stop.set()
		try:
			if self.proc:
				self.proc.terminate()
				try: self.proc.wait(timeout=1.0)
				except subprocess.TimeoutExpired: self.proc.kill()
		except Exception: pass
		if self._thr and self._thr.is_alive():
			try: self._thr.join(timeout=1.0)
			except Exception: pass

		if not self._samples:
			# single-shot fallback to avoid NaN
			try:
				tegra = shutil.which("tegrastats") or "/usr/bin/tegrastats"
				out = subprocess.check_output([tegra, "--interval", "1000"], timeout=1.2, text=True)
				for line in out.splitlines():
					rails = self._parse_line(line)
					p = self._pick_power(rails)
					if p is not None:
						self._samples.append((time.time(), p))
						break
			except Exception: pass

	def mean_power(self):
		with self._lock:
			if not self._samples: return float("nan")
			return float(sum(p for _,p in self._samples)/len(self._samples))
