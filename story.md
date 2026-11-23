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
  (`connection_string`). A small step now restores the original spelling.
- Scheduled workflows in a public repo switch themselves off after 60 days with no commits.
  A weekly job re-enables it.
- The generators were built for a web server, so the slow parts run in background threads and
  the request returns at once. In a job, returning at once means the work dies with the
  process. The job runner now waits for exactly those threads.
- Two systems generating the same day at the same time would double the cost, so Actions only
  publishes until a single switch says it's its turn.

Numbers for the post: 7,607 published files (about 140 MB) cover the full archive, from 983
English front pages to 579 comic pages and 3,929 research entries.
