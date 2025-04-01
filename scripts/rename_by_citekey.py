import os
import re
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def extract_md_name_from_file_field(file_field):
    """Extract the markdown filename from the file field in BibTeX."""
    if not file_field:
        return None
    
    # The file field is in the format: {/path/to/file/Author - Year - Title.pdf}
    # Extract just the filename part
    match = re.search(r'/([^/]+)\.pdf', file_field)
    if match:
        filename = match.group(1)
        return f"{filename}.md"
    return None

def parse_bibtex(bibtex_file):
    """Parse the BibTeX file and extract mapping of filenames to citation keys."""
    filename_to_citekey = {}
    current_entry = None
    current_citekey = None
    
    with open(bibtex_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Start of a new entry
            if line.startswith('@'):
                match = re.search(r'@\w+{([^,]+),', line)
                if match:
                    current_citekey = match.group(1)
                    current_entry = {}
            
            # End of an entry
            elif line == '}' and current_citekey and current_entry is not None:
                if 'file' in current_entry:
                    md_name = extract_md_name_from_file_field(current_entry['file'])
                    if md_name:
                        filename_to_citekey[md_name] = f"{current_citekey}.md"
                current_entry = None
                current_citekey = None
            
            # Field in the entry
            elif current_entry is not None and '=' in line:
                field_match = re.match(r'(\w+)\s*=\s*{(.+)}', line)
                if field_match:
                    field_name = field_match.group(1)
                    field_value = field_match.group(2)
                    current_entry[field_name] = field_value
                else:
                    # Handle multi-line fields
                    field_match = re.match(r'(\w+)\s*=\s*{(.+)', line)
                    if field_match:
                        field_name = field_match.group(1)
                        field_value = field_match.group(2)
                        current_entry[field_name] = field_value
                    elif 'file' in current_entry and not current_entry['file'].endswith('}'):
                        # Continue adding to file field
                        current_entry['file'] += line
    
    return filename_to_citekey

def main():
    parser = argparse.ArgumentParser(description='Rename markdown files based on BibTeX citation keys')
    parser.add_argument('--input_dir', required=True, help='Directory containing markdown files')
    parser.add_argument('--output_dir', required=True, help='Directory for renamed markdown files')
    parser.add_argument('--bibtex_file', required=True, help='BibTeX file with citation keys')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse BibTeX file for filename to citekey mapping
    filename_to_citekey = parse_bibtex(args.bibtex_file)
    
    # Recursively find all markdown files
    input_path = Path(args.input_dir)
    md_files = []
    for md_path in input_path.rglob('*.md'):
        # Skip meta.md files
        if md_path.name == 'meta.md':
            continue
        # Get relative path from input directory
        rel_path = md_path.relative_to(input_path)
        # If file is in a subdirectory, use the directory name as the original filename
        if len(rel_path.parts) > 1:
            md_files.append((str(md_path), rel_path.parts[0] + '.md'))
        else:
            md_files.append((str(md_path), rel_path.name))
    
    # Special case patterns for problematic filenames
    special_cases = {
        # Add cases as needed
    }
    
    # Known citation keys for papers that might not have exact file matches
    author_year_to_citekey = {
        ('bivall', '2010'): 'bivallHapticJustNoticeable2010',
        # Add more as needed
    }
    
    # Process each markdown file
    renamed_count = 0
    not_found_count = 0
    special_case_count = 0
    
    for md_path, original_name in md_files:
        # Check for direct special case
        if original_name in special_cases:
            output_path = os.path.join(args.output_dir, special_cases[original_name])
            shutil.copy2(md_path, output_path)
            print(f"Copied {original_name} to {special_cases[original_name]} (special case)")
            special_case_count += 1
            continue
        
        # Check for pattern-based special case
        special_case_match = False
        for pattern, replacement in special_cases.items():
            if pattern.startswith('r\'') and re.search(pattern[2:-1], original_name):
                output_path = os.path.join(args.output_dir, replacement)
                shutil.copy2(md_path, output_path)
                print(f"Copied {original_name} to {replacement} (special case pattern)")
                special_case_count += 1
                special_case_match = True
                break
        
        if special_case_match:
            continue
        
        # Try to find a matching filename in the BibTeX mapping
        if original_name in filename_to_citekey:
            new_name = filename_to_citekey[original_name]
            output_path = os.path.join(args.output_dir, new_name)
            shutil.copy2(md_path, output_path)
            print(f"Copied {original_name} to {new_name}")
            renamed_count += 1
            continue
        
        # Try author+year matching from filename
        found_match = False
        parts = original_name.split(' - ')
        if len(parts) >= 2:
            author = parts[0].lower().split()[0]  # Get first word of author
            year_match = re.search(r'(\d{4})', parts[1])
            if year_match:
                year = year_match.group(1)
                key = (author, year)
                if key in author_year_to_citekey:
                    citekey = f"{author_year_to_citekey[key]}.md"
                    output_path = os.path.join(args.output_dir, citekey)
                    shutil.copy2(md_path, output_path)
                    print(f"Copied {original_name} to {citekey} (author-year match)")
                    renamed_count += 1
                    found_match = True
                    continue
        
        # Try to find a close match by comparing author and year
        if not found_match:
            for bib_file, citekey in filename_to_citekey.items():
                bib_parts = bib_file.split(' - ')
                md_parts = original_name.split(' - ')
                
                if len(bib_parts) >= 2 and len(md_parts) >= 2:
                    # Compare first author's last name and year
                    bib_author = bib_parts[0].lower().split()[-1]  # Last word of author field
                    md_author = md_parts[0].lower().split()[-1]
                    
                    bib_year = bib_parts[1]
                    md_year = md_parts[1]
                    
                    # Allow for partial author name matches (e.g., "Bivall" matches "Bivall and Forsell")
                    if (bib_author in md_parts[0].lower() or md_author in bib_parts[0].lower()) and bib_year == md_year:
                        output_path = os.path.join(args.output_dir, citekey)
                        shutil.copy2(md_path, output_path)
                        print(f"Copied {original_name} to {citekey} (fuzzy author-year match)")
                        renamed_count += 1
                        found_match = True
                        break
        
        if not found_match:
            print(f"Could not find matching citation key for {original_name}")
            not_found_count += 1
    
    print(f"\nSummary:")
    print(f"  Renamed: {renamed_count}")
    print(f"  Special cases: {special_case_count}")
    print(f"  Not found: {not_found_count}")
    print(f"  Total: {renamed_count + special_case_count + not_found_count}")

if __name__ == "__main__":
    main() 