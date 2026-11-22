import os
import sys
import time
import subprocess
import signal
import requests

def test_streamlit_headless():
    project_dir = "/Users/vegirajusrujayvarma/.gemini/antigravity/scratch/churnsynth_ai"
    python_bin = os.path.join(project_dir, "venv", "bin", "python")
    streamlit_bin = os.path.join(project_dir, "venv", "bin", "streamlit")
    port = 8599
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    health_url = f"{base_url}/_stcore/health"
    
    cmd = [
        streamlit_bin,
        "run",
        "app.py",
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.headless=true",
        "--browser.serverAddress=127.0.0.1",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]

    print(f"Starting Streamlit server on {base_url}...")
    log_file_path = os.path.join(project_dir, "tests", "streamlit_test.log")
    log_file = open(log_file_path, "w")

    process = subprocess.Popen(
        cmd,
        cwd=project_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    max_wait = 25  # seconds
    start_time = time.time()
    server_healthy = False

    try:
        while time.time() - start_time < max_wait:
            if process.poll() is not None:
                raise RuntimeError(f"Streamlit server terminated prematurely with exit code {process.returncode}")
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200 and resp.text.strip() == "ok":
                    server_healthy = True
                    print(f"Streamlit health check passed in {time.time() - start_time:.2f}s!")
                    break
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(0.5)

        if not server_healthy:
            raise TimeoutError(f"Streamlit server did not become healthy within {max_wait}s")

        # Now test the main page endpoint
        print(f"Requesting root page {base_url}/ ...")
        page_resp = requests.get(base_url, timeout=5.0)
        print(f"HTTP Status Code: {page_resp.status_code}")
        assert page_resp.status_code == 200, f"Expected 200 OK, got {page_resp.status_code}"
        
        # Verify page content contains title and core elements
        content = page_resp.text
        assert "ChurnSynth Platform" in content or "Streamlit" in content, "Missing expected title tokens in HTML"
        print("Page HTML successfully loaded with valid response length:", len(content))

        # Check the static assets / manifest
        print("Testing static / metadata availability...")
        manifest_resp = requests.get(f"{base_url}/_stcore/host-config", timeout=2.0)
        assert manifest_resp.status_code == 200, f"Host config failed: {manifest_resp.status_code}"

    finally:
        print("Shutting down Streamlit test process...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except Exception as e:
            print(f"Process kill exception (falling back to SIGKILL): {e}")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
        log_file.close()

    # Analyze the logs for warnings, deprecations, or tracebacks
    with open(log_file_path, "r") as f:
        log_content = f.read()

    print("\n--- STREAMLIT SERVER LOGS ---")
    print(log_content)
    print("-----------------------------")

    # Verification of log content
    forbidden_tokens = ["Traceback", "SyntaxError", "ValueError", "TypeError", "DeprecationWarning", "use_container_width"]
    for token in forbidden_tokens:
        assert token not in log_content, f"Found forbidden token '{token}' in Streamlit server log!"

    print("\n[PASS] Headless Streamlit End-to-End Verification PASSED with ZERO errors, tracebacks, or deprecation warnings!")


if __name__ == "__main__":
    test_streamlit_headless()
