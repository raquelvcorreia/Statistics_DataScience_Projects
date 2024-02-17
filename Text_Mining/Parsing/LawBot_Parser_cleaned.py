from cobalt import AkomaNtosoDocument
import pandas as pd
import numpy as np

file = open(r'C:\Users\raque\Downloads\test_lawbot.xml', encoding="utf8")
xml_data = file.read()
file.close()

akn_doc_en = AkomaNtosoDocument(xml_data)

# Xml for the german version
#"https://www.fedlex.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/27/317_321_377/20240101/de/xml/fedlex-data-admin-ch-eli-cc-27-317_321_377-20240101-de-xml-3.xml"

file = open(r'C:\Users\raque\Downloads\test_lawbot_220_de.xml', encoding="utf8")
xml_data_de = file.read()
file.close()

akn_doc_de = AkomaNtosoDocument(xml_data_de)


def process_article(df, article, section_titles):
    # Check article
    if article is None:
        return df

    article_id = article.attrib["eId"]
    article_title = ' '.join([x.text.strip() for x in article.num.getchildren()]).strip()

    # this excludes articles with no paragraphs like Art. 40g
    if not hasattr(article, 'paragraph') or len(article.paragraph) == 0:
        return df

    article_paragraph_num = len(article.paragraph)
    paragraphs_txt = []
    for paragraph in article.paragraph:
        if hasattr(paragraph.content, 'p'):
            paragraphs_txt += [x.text.strip() for x in paragraph.content.p]
        if hasattr(paragraph.content, 'blockList') and hasattr(paragraph.content.blockList, 'listIntroduction'):
            extra_text = paragraph.content.blockList.listIntroduction.text
            paragraphs_txt += [extra_text.strip()]
            if hasattr(paragraph.content.blockList.item, 'p'):
                article_paragraph_num += len(paragraph.content.blockList.item)
                paragraphs_txt += [x.text.strip() for x in paragraph.content.blockList.item.p]
            if hasattr(paragraph.content.blockList, 'item'):
                if hasattr(paragraph.content.blockList.item, 'blockList'):
                    extra_text = paragraph.content.blockList.item.blockList.listIntroduction.text
                    paragraphs_txt += [extra_text.strip()]
                    if hasattr(paragraph.content.blockList.item.blockList, 'item') and hasattr(paragraph.content.blockList.item.blockList, 'p'):
                        article_paragraph_num += len(paragraph.content.blockList.item.blockList.item)
                        paragraphs_txt += [x.text.strip() for x in paragraph.content.blockList.item.blockList.item.p]
        if hasattr(paragraph.content, 'p') and hasattr(paragraph.content.p, 'inline'):
            paragraphs_txt += [x.text.strip() for x in paragraph.content.p.inline]

    paragraphs_txt = ','.join(paragraphs_txt)

    new_row = pd.DataFrame({'Article_Name': article_title, 'SectionTitles': [section_titles], 'Num_Paragraphs': article_paragraph_num, 'Merged_Text': paragraphs_txt})
    df = pd.concat([df, new_row], ignore_index=True)
    return df


def process_sections(df, sections, section_titles):
    for section in sections:
        section_title = str(section.num).strip()
        df = find_articles(df, section, section_titles + [section_title])

    return df


def find_articles(df, parent, section_titles):
    # Check parent
    if parent is None:
        return df

    # Find all articles that are children of this parent
    if hasattr(parent, 'article'):
        for article in parent.article:
            df = process_article(df, article, section_titles)

    # Find all sections of type part
    if hasattr(parent, 'part'):
        # print('FOUND PARTS')
        df = process_sections(df, parent.part, section_titles)

    # Find all sections of type title
    if hasattr(parent, 'title'):
        df = process_sections(df, parent.title, section_titles)

    # Find all sections of type chapter
    if hasattr(parent, 'chapter'):
        df = process_sections(df, parent.chapter, section_titles)

    # Find all sections of type level
    if hasattr(parent, 'level'):
        df = process_sections(df, parent.level, section_titles)

    return df


# Create Dataframe to store the data
# en
df_data_compiled = pd.DataFrame(columns=['Article_Name', 'SectionTitles', 'Num_Paragraphs', 'Merged_Text'])
df_data_compiled = find_articles(df_data_compiled, akn_doc_en.root.act.body, [])
# de
df_data_compiled_de = pd.DataFrame(columns=['Article_Name', 'SectionTitles', 'Num_Paragraphs', 'Merged_Text'])
df_data_compiled_de = find_articles(df_data_compiled_de, akn_doc_de.root.act.body, [])

# Save Dataframe to file
df_data_compiled.to_json("cleaned_pflichten_des_arbeitsgebers_full_en.json")
df_data_compiled.to_csv("cleaned_pflichten_des_arbeitsgebers_full.en.csv", sep=";")

df_data_compiled_de.to_json("cleaned_pflichten_des_arbeitsgebers_full_de.json")
df_data_compiled_de.to_csv("cleaned_pflichten_des_arbeitsgebers_full_de.csv", sep=";")

