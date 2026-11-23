# The story of SandBox

Notes for blog posts and social posts about building SandBox (sandboxes.live). Newest first.

## Nov 2026: moving off the App Service

SandBox is six AI projects in one Flask app: a newspaper from tomorrow, a daily AI research
blog, a comic with running story arcs, an "AI government" TV channel, an agent chipping away at
open maths problems, and a dashboard for a trading bot. For two years it all ran on one Azure
App Service, about $123 a month, plus five Logic Apps that poked it on a timer so each day's
content got made.

The thing is, the server mostly sat there. Every page is the same for every visitor. The only
real work happens a few times a day, when a generator runs for anywhere from two minutes to
half an hour. Paying for a box that's awake all month to do that felt silly.

So the plan was to keep the expensive and stateful bits (Azure OpenAI through APIM, and the
storage account with two years of archives) and drop only the server:

- GitHub Pages serves the pages.
- A Cloudflare Worker answers the data requests the pages make, on the same domain, so not a
  single line of the front end had to change.
- GitHub Actions runs the generators on the same schedule the Logic Apps had.

The part I liked most: the Worker was tested against the old Flask app URL by URL, same
bodies, same status codes, same headers, 28 out of 28. That's what lets you swap the engine
without anyone noticing.

Things that bit along the way:

- GitHub stores secret names in upper case. Half of our settings are lower case
  (`connection_string`). My first fix handed the job all secrets at once and restored the
  spelling in a script. GitHub took one look and marked the workflow "may be malicious",
  because dumping every secret is exactly what secret-stealing workflows do. Fair enough. Now
  the workflow names each secret on its own line.
- Scheduled workflows in a public repo switch themselves off after 60 days with no commits.
  A weekly job re-enables it.
- The generators were built for a web server, so the slow parts run in background threads and
  the request returns at once. In a job, returning at once means the work dies with the
  process. The job runner now waits for exactly those threads.
- Two systems generating the same day at the same time would double the cost, so Actions only
  publishes until a single switch says it's its turn.

- The trading dashboard's price charts went blank about half the time once they were served
  from Cloudflare. The exchange we read candles from rate-limits by IP, and a Worker shares its
  outgoing IPs with a huge number of other Workers. The old server had an IP all to itself, so
  this never showed up before. The fix: retry briefly, fall back to a second exchange, and keep
  the last good chart for a few minutes. Thirty calls in a row, thirty charts.

The cut-over itself, on Nov 23: test everything on beta.sandboxes.live first, compare the new
data URLs against the live old server one by one (28 of 28 identical, trAIde matched on shape
because its numbers move every few seconds), switch off the five Logic Apps, let Actions take
over generation, then point the domain at GitHub Pages through Cloudflare. Nobody visiting
would have noticed a thing. The last bit of tidying was deleting the beta hostname, because a
DNS record that points at GitHub Pages without a site claiming it is an open door: anyone could
put their own Pages site on it.

Numbers for the post: 7,607 published files (about 140 MB) cover the full archive, from 983
English front pages to 579 comic pages and 3,929 research entries.
