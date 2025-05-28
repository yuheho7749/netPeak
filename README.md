# netPeek

**netPeek** is a machine‑learning–powered “speed‑test shortcut.”  
Instead of saturating the network for tens of seconds the way conventional tests (e.g., Ookla or iPerf) do, netPeek watches only the first *k* epochs (collections of packets in a given period of time) of the network flow, extracts light‑weight timing and size features, and immediately predicts the final download throughput. By ending the test early once a confident prediction is available, we free up bandwidth on constrained links and make large‑scale measurement campaigns less intrusive.

Under the hood, we treat throughput estimation as a regression problem. Given a short prefix of each speed‑test trace, our models output a Mbps prediction and are trained to minimise root‑mean‑squared error against ground‑truth speeds from a NetReplica dataset, where NDT download tests are conducted under a variety of network conditions. We extract TCP_INFO statistics from measurements during the NDT test as our feature set.

Once we have a regression model trained, we take the predicted values for our dataset and compare them with the ground-truth speeds, labeling a test as capable of being terminated early if the prediction is close to the final recorded value. This augmented dataset serves as the basis for a classification problem, where we try to determine whether or not to terminate mid-test. We use TRUSTEE‑style post‑hoc reports to verify that each model’s decisions rest on sensible cues.
