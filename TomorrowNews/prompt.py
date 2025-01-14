from datetime import datetime, timedelta
from TomorrowNews.azurestorage import get_row, insert_history
from TomorrowNews.graph import news_graph
from utils import get_flat_date_hour

def gettomorrownews(parsed_date):
    flat_date_hour = get_flat_date_hour(parsed_date)
    timestamp = datetime.utcnow()
    next_day = timestamp + timedelta(days=1)
    if lasthournews := get_row(flat_date_hour):
        return lasthournews["html_content"], lasthournews.metadata["timestamp"]
    flat_date_hour = get_flat_date_hour()
    if parsed_date is not None and (lasthournews := get_row(flat_date_hour)):
        return lasthournews["html_content"], lasthournews.metadata["timestamp"] 
    
    for event in news_graph.stream({"messages": [("system", f"""
                                                Using today’s ({timestamp.strftime('%Y-%m-%d')}) actual newspaper as a foundation, \
                                                apply reasoning and analysis to predict future events. \
                                                Create the next day’s ({next_day.strftime('%Y-%m-%d')}) edition of 'Tomorrow News,' \
                                                complete with imaginative yet plausible headlines and stories. \
                                                Avoid simply continuing or expanding on today’s news—instead, \
                                                focus on predicting the next events and news that could arise as consequences of current happenings or emerge unexpectedly. \
                                                Make it feel like a genuine glimpse into the future of politics, geopolitics, economy, events, Culture, \
                                                Environment, Technology, Health, Security, Education, Science, Energy, Trade, Human Rights, Diplomacy, Military, \
                                                Infrastructure, Agriculture, Transportation, Media, Religion, Demographics, Finance, Law, Tourism, Sports, Migration and what come next!\
                                                    
                                                Next, design an HTML page for the newspaper. The layout should resemble a professional newspaper, optimized for both desktop and mobile screens. Ensure that the design includes:

                                                A clear newspaper header with the title Tomorrow News.
                                                A two-column layout on the left and right sides of the page, with a wide central column for the main content.
                                                Visually appealing headlines in the central column with a proper headline style, including a large main headline at the top.
                                                Use the content created before and fill all the columns.
                                                A well-balanced font selection with readable sizes, appropriate contrast, and a cohesive color scheme.
                                                A clean, minimalistic border around the entire page for a polished look.
                                                Each story should be long enough to fill the column space like a real newspaper article—meaning substantial content for each story to resemble actual newspaper columns in length.
                                                Images for news stories: Use the image tool to create realistic photos that complement the headlines and add them as appropriate (with image URLs integrated into the HTML).
                                                to create the best photo, think as a newspaper photographer and describe the photo with details and tell the tool explicitly to create a realistic photo.
                                                Be aware that the image tool has content filtering and cannot create any image, try to create images that are not going to filtered, an in case of an error raised by content filtering retry another photo instead of that.
                                                The layout should avoid unnecessary gaps and ensure that content is well-aligned and fits seamlessly into the space.
                                                Prioritize responsive design, so the layout adapts beautifully to both desktop and mobile screens.
                                                After creating the visual design and content, ensure the HTML is well-structured and ready to be rendered correctly by a browser, making it appear as a genuine newspaper page, with functional columns, images, and headings.

                                                Output: Pure HTML code without anything extra! (not even ```html at start and ``` at the end) that includes all of the above elements in a clean, readable, and responsive format. your response will be parsed directly in a browser, so should be rendered correctly as an HTML
                                                  """)]}):
        print("event: ", event)
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
    content = value["messages"][-1].content
    insert_history(rowkey=flat_date_hour, html_content=content)
    return content, timestamp