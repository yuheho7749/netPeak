# netPeek

**netPeek** is a machine‑learning–powered “speed‑test shortcut.”  
Instead of saturating the network for tens of seconds the way conventional tests (e.g., Ookla or iPerf) do, netPeek watches only the first *k* packets of the flow, extracts light‑weight timing and size features, and immediately predicts the final download and upload rates. By ending the test early once a confident prediction is available, we free up bandwidth on constrained links and make large‑scale measurement campaigns less intrusive.

Under the hood, we treat throughput estimation as a regression problem.  Given a short prefix of each speed‑test trace, our models output a Mbps prediction and are trained to minimise mean‑absolute and/or root‑mean‑squared error against ground‑truth speeds from public datasets such as M‑Lab NDT and FCC Measuring Broadband America.  We sweep several values of *k* to quantify the trade‑off between test length and accuracy, and we use TRUSTEE‑style post‑hoc reports to verify that the model’s decisions rest on sensible packet‑level cues.

netPeek's advantage lies in providing network measurements that finish in milliseconds rather than minutes. Reproducible experiments and a demo are on their way.
