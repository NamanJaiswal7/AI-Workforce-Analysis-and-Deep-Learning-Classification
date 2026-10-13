import pandas as pd
import os

ISCO_DATA = [
    {"code": "11", "title": "Chief Executives, Senior Officials and Legislators", "description": "Formulate and review policies and plan, direct, coordinate and evaluate the overall activities of enterprises, governments and other organizations."},
    {"code": "12", "title": "Administrative and Commercial Managers", "description": "Plan, organize, direct, control and coordinate the administrative and commercial activities of an enterprise or organization."},
    {"code": "13", "title": "Production and Specialized Services Managers", "description": "Plan, direct and coordinate the production of goods and the provision of specialized professional and technical services."},
    {"code": "14", "title": "Hospitality, Retail and Other Services Managers", "description": "Plan, organize, direct, control and coordinate the provision of services in establishments such as hotels, restaurants, shops, etc."},
    {"code": "21", "title": "Science and Engineering Professionals", "description": "Conduct research, improve or develop concepts, theories and operational methods, or apply scientific knowledge relating to fields such as physics, astronomy, meteorology, chemistry, geophysics, geology, mathematics, statistics, computing, architecture, engineering, and technology."},
    {"code": "22", "title": "Health Professionals", "description": "Conduct research, improve or develop concepts, theories and operational methods, and apply scientific knowledge relating to medicine, nursing, dentistry, veterinary medicine, pharmacy, and promotion of health."},
    {"code": "23", "title": "Teaching Professionals", "description": "Teach the theory and practice of one or more disciplines at different educational levels, conduct research and improve or develop concepts, theories and operational methods pertaining to their particular discipline, and prepare scholarly papers and books."},
    {"code": "24", "title": "Business and Administration Professionals", "description": "Perform analytical, conceptual and practical tasks to provide services in financial matters, human resource development, public relations, marketing and sales in the commercial, public and private sectors."},
    {"code": "25", "title": "Information and Communications Technology Professionals", "description": "Conduct research, plan, design, write, test, provide advice and improve information technology systems, hardware, software and related concepts for specific applications."},
    {"code": "26", "title": "Legal, Social and Cultural Professionals", "description": "Conduct research, improve or develop concepts, theories and operational methods, or apply knowledge relating to the law, economics, social sciences, religion, visual and performing arts, sports and entertainment."},
    {"code": "31", "title": "Science and Engineering Associate Professionals", "description": "Perform technical tasks connected with research and the application of scientific or engineering concepts and operational methods."},
    {"code": "32", "title": "Health Associate Professionals", "description": "Perform technical tasks connected with research and the application of scientific or medical knowledge to the provision of health care, treatment and related services."},
    {"code": "33", "title": "Business and Administration Associate Professionals", "description": "Perform mostly technical tasks connected with the practical application of knowledge relating to financial accounting and transaction processing, mathematical calculations, human resource development, selling and buying financial instruments, specialized secretarial tasks, and enforcing or applying relevant government rules."},
    {"code": "34", "title": "Legal, Social, Cultural and Related Associate Professionals", "description": "Perform technical tasks connected with the practical application of knowledge relating to legal services, social work, culture, food preparation, sports and religion."},
    {"code": "35", "title": "Information and Communications Technicians", "description": "Provide support for the day-to-day running of computer systems, communications systems and networks, and perform technical tasks related to telecommunications, broadcasting of image and sound, and other types of telecommunication signals on land, sea or in aircraft."},
    {"code": "41", "title": "General and Keyboard Clerks", "description": "Record, organize, store, compute and retrieve information, and perform a number of clerical duties especially in connection with money-handling operations, travel arrangements, requests for information, and appointments."},
    {"code": "42", "title": "Customer Services Clerks", "description": "Deal directly with clients in connection with money-handling operations, travel arrangements, requests for information, and appointments."},
    {"code": "43", "title": "Numerical and Material Recording Clerks", "description": "Obtain, compile and compute accounting, bookkeeping, statistical, financial, and other numerical data, and take charge of cash transactions incidental to business matters."},
    {"code": "44", "title": "Other Clerical Support Workers", "description": "Sort and deliver mail, file documents, prepare data for processing, maintain personnel records, and carry out other specialized clerical duties."},
    {"code": "51", "title": "Personal Service Workers", "description": "Provide personal and protective services related to travel, housekeeping, catering, personal care, or protection against fire and unlawful acts."},
    {"code": "52", "title": "Sales Workers", "description": "Sell goods and services in retail or wholesale establishments, and provide services to customers."},
    {"code": "53", "title": "Personal Care Workers", "description": "Provide care, supervision and assistance for children, patients and elderly, convalescent or disabled persons in institutional and residential settings."},
    {"code": "54", "title": "Protective Services Workers", "description": "Protect individuals and property against fire and other hazards, maintain law and order and enforce laws and regulations."},
    {"code": "61", "title": "Market-oriented Skilled Agricultural Workers", "description": "Plan, organize and perform farming operations to grow and harvest crops, and to breed and raise animals for sale or delivery on a regular basis to wholesale buyers, marketing organizations or at markets."},
    {"code": "62", "title": "Market-oriented Skilled Forestry, Fishery and Hunting Workers", "description": "Plan, organize and perform operations to cultivate, conserve and exploit forests, and to breed or catch fish, and to hunt or trap animals for sale or delivery on a regular basis to wholesale buyers, marketing organizations or at markets."},
    {"code": "63", "title": "Subsistence Farmers, Fishers, Hunters and Gatherers", "description": "Grow and harvest field or tree and shrub crops, grow vegetables and fruit, gather wild fruits, medicinal and other plants, tend or hunt animals, catch fish and gather various forms of aquatic life in order to provide food, shelter and, in some cases, a minimum of cash income for themselves and their households."},
    {"code": "71", "title": "Building and Related Trades Workers, excluding Electricians", "description": "Construct, maintain and repair buildings, erect and repair foundations, walls and structures of brick, stone and similar materials, shape and finish stone for building and other purposes."},
    {"code": "72", "title": "Metal, Machinery and Related Trades Workers", "description": "Make and repair tools, equipment and other metal articles, build structure framework, shape and finish metal, fit, maintain and repair industrial machinery, engines, vehicles and other electrical and electronic equipment."},
    {"code": "73", "title": "Handicraft and Printing Workers", "description": "Combine artistic and manual skills to design, produce, maintain and decorate precision instruments, musical instruments, jewelry and other precious metals, pottery, porcelain and glassware, as well as articles made of wood, textile, leather or other materials."},
    {"code": "74", "title": "Electrical and Electronic Trades Workers", "description": "Fit, adjust, maintain and repair electrical and electronic equipment, wiring, and systems."},
    {"code": "75", "title": "Food Processing, Woodworking, Garment and Other Craft and Related Trades Workers", "description": "Treat and process agricultural and marine raw materials to make food and other products, make and repair goods made of wood, textiles, fur, leather, or other materials."},
    {"code": "81", "title": "Stationary Plant and Machine Operators", "description": "Operate and monitor industrial plant and machinery, equipment and automated or semi-automated systems for processing of materials, production of goods, and control of machinery."},
    {"code": "82", "title": "Assemblers", "description": "Assemble components into semi-finished or finished products according to strict procedures."},
    {"code": "83", "title": "Drivers and Mobile Plant Operators", "description": "Drive and tend trains and motor vehicles, or drive, operate and monitor industrial and agricultural machinery and equipment, or execute other tasks on board ships and other water-borne craft."},
    {"code": "91", "title": "Cleaners and Helpers", "description": "Sweep, vacuum, wash and polish floors, furniture and other objects in buildings, hotels, offices, hospitals, and other establishments, or perform simple tasks in preparing food, or clean vehicles, windows and other surfaces."},
    {"code": "92", "title": "Agricultural, Forestry and Fishery Labourers", "description": "Perform simple and routine tasks in the production of crops and livestock, in forestry and fishery operations."},
    {"code": "93", "title": "Labourers in Mining, Construction, Manufacturing and Transport", "description": "Perform simple and routine tasks in connection with mining, construction, manufacturing and transport operations."},
    {"code": "94", "title": "Food Preparation Assistants", "description": "Prepare and cook to order a small variety of pre-cooked food or beverages, clear tables, clean kitchen areas and wash dishes."},
    {"code": "95", "title": "Street and Related Sales and Service Workers", "description": "Sell goods or provide services in streets and public places, or distribute leaflets and free newspapers."},
    {"code": "96", "title": "Refuse Workers and Other Elementary Workers", "description": "Collect garbage and recyclable materials, sweep streets and similar places, perform various simple tasks for the maintenance and upkeep of buildings and grounds, and do odd jobs."},
    {"code": "01", "title": "Commissioned Armed Forces Officers", "description": "Commissioned armed forces officers provide leadership and management to organizational units in the armed forces and/or perform similar tasks to those performed in a variety of civilian occupations outside the armed forces."},
    {"code": "02", "title": "Non-commissioned Armed Forces Officers", "description": "Non-commissioned armed forces officers enforce military discipline and supervise the activities of those employed in Armed forces occupations, other ranks, and/or perform similar tasks to those performed in a variety of civilian occupations outside the armed forces."},
    {"code": "03", "title": "Armed Forces Occupations, Other Ranks", "description": "Armed forces occupations, other ranks include all conscripts and non-commissioned volunteers who perform tasks relating to the defence of their country."}
]

def load_isco_data(filepath):
    """Generate and structure the ISCO-08 classification hierarchy."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    isco_df = pd.DataFrame(ISCO_DATA)
    isco_df.to_csv(filepath, index=False)
    
    print(f"ISCO-08 codes loaded and generated: {len(isco_df)} at 2-digit level")
    
    return isco_df

if __name__ == "__main__":
    load_isco_data("data/raw/isco08_en.csv")
