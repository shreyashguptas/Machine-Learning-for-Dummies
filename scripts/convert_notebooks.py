import os
import nbformat
from nbconvert import MarkdownExporter

def convert_notebook_to_markdown(notebook_path, output_dir):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Create markdown exporter
    markdown_exporter = MarkdownExporter()
    
    # Convert to markdown
    (body, resources) = markdown_exporter.from_notebook_node(nb)
    
    # Get the notebook filename without extension
    filename = os.path.splitext(os.path.basename(notebook_path))[0]
    
    # Create markdown file path
    markdown_path = os.path.join(output_dir, f"{filename}.md")
    
    # Add Jekyll front matter
    front_matter = f"""---
layout: chapter
title: {filename.replace('-', ' ')}
---

"""
    body = front_matter + body
    
    # Write the markdown file
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(body)

def main():
    # Create _chapters directory if it doesn't exist
    os.makedirs('_chapters', exist_ok=True)
    
    # Convert all notebooks in chapters directory
    for filename in os.listdir('chapters'):
        if filename.endswith('.ipynb'):
            notebook_path = os.path.join('chapters', filename)
            convert_notebook_to_markdown(notebook_path, '_chapters')

if __name__ == '__main__':
    main() 