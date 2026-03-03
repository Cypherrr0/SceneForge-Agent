#!/usr/bin/env python3
"""
Parse Blender Python API HTML documentation and extract structured information for RAG.
Focuses on bpy.types.* files only.
"""

import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from tqdm import tqdm


class BpyTypesDocParser:
    def __init__(self, docs_dir: str, output_file: str):
        self.docs_dir = Path(docs_dir)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'total_classes': 0,
            'total_methods': 0,
            'total_attributes': 0
        }
        
        self.failed_files = []
    
    def parse_all(self):
        """Parse all bpy.types.* HTML files"""
        # Find all bpy.types.* HTML files
        html_files = list(self.docs_dir.glob("bpy.types.*.html"))
        self.stats['total_files'] = len(html_files)
        
        print(f"Found {len(html_files)} bpy.types.* HTML files\n")
        
        all_classes = []
        
        # Process each file with progress bar
        for html_file in tqdm(html_files, desc="Parsing documentation", unit="file"):
            try:
                class_info = self.parse_types_file(html_file)
                if class_info:
                    all_classes.append(class_info)
                    self.stats['processed'] += 1
                    self.stats['total_classes'] += 1
                    self.stats['total_methods'] += len(class_info.get('methods', []))
                    self.stats['total_attributes'] += len(class_info.get('attributes', []))
            except Exception as e:
                self.failed_files.append({
                    'file': html_file.name,
                    'error': str(e)
                })
                self.stats['failed'] += 1
                tqdm.write(f"⚠️  Failed to parse: {html_file.name} - {str(e)}")
        
        # Save results
        self.save_json(all_classes)
        self.print_stats()
    
    def parse_types_file(self, html_file: Path) -> Optional[Dict[str, Any]]:
        """Parse a single bpy.types.* file"""
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Find the main content section
        main_content = soup.find('article', id='furo-main-content')
        if not main_content:
            return None
        
        # Extract page title (e.g., "BlendDataCurves(bpy_struct)")
        h1 = main_content.find('h1')
        if not h1:
            return None
        
        page_title = self.clean_text(h1.get_text())
        
        # Find the class definition
        class_dl = main_content.find('dl', class_='py class')
        if not class_dl:
            return None
        
        # Parse class information
        class_dt = class_dl.find('dt', class_='sig sig-object py')
        if not class_dt:
            return None
        
        class_id = class_dt.get('id', '')
        class_name = self.extract_class_name(class_dt)
        base_class = self.extract_base_class(class_dt)
        
        # Extract class description
        class_dd = class_dl.find('dd', recursive=False)
        description = ''
        if class_dd:
            desc_p = class_dd.find('p', recursive=False)
            if desc_p:
                description = self.clean_text(desc_p.get_text())
        
        # Extract methods
        methods = []
        if class_dd:
            for method_dl in class_dd.find_all('dl', class_='py method', recursive=False):
                method_info = self.parse_method(method_dl)
                if method_info:
                    methods.append(method_info)
        
        # Extract attributes/properties
        attributes = []
        if class_dd:
            for attr_dl in class_dd.find_all('dl', class_='py attribute', recursive=False):
                attr_info = self.parse_attribute(attr_dl)
                if attr_info:
                    attributes.append(attr_info)
        
        # Build complete class information
        class_info = {
            'page_title': page_title,
            'class_id': class_id,
            'class_name': class_name,
            'base_class': base_class,
            'description': description,
            'methods': methods,
            'attributes': attributes,
            'source_file': html_file.name
        }
        
        return class_info
    
    def parse_method(self, method_dl) -> Optional[Dict[str, Any]]:
        """Parse a method definition"""
        try:
            method_dt = method_dl.find('dt', class_='sig sig-object py')
            if not method_dt:
                return None
            
            method_id = method_dt.get('id', '')
            
            # Extract method name
            method_name_span = method_dt.find('span', class_='sig-name descname')
            if not method_name_span:
                return None
            method_name = self.clean_text(method_name_span.get_text())
            
            # Extract method signature (full text from dt)
            signature = self.clean_text(method_dt.get_text())
            
            # Check if it's a classmethod
            is_classmethod = bool(method_dt.find('em', class_='property', string=re.compile('classmethod')))
            
            # Extract method description
            method_dd = method_dl.find('dd')
            description = ''
            if method_dd:
                desc_p = method_dd.find('p', recursive=False)
                if desc_p:
                    description = self.clean_text(desc_p.get_text())
            
            # Extract parameters
            parameters = []
            returns = {}
            if method_dd:
                field_list = method_dd.find('dl', class_='field-list')
                if field_list:
                    # Parse parameters
                    param_section = self.find_field_section(field_list, 'Parameters')
                    if param_section:
                        parameters = self.parse_parameters(param_section)
                    
                    # Parse return information
                    returns_section = self.find_field_section(field_list, 'Returns')
                    return_type_section = self.find_field_section(field_list, 'Return type')
                    
                    if returns_section or return_type_section:
                        returns = {
                            'description': self.clean_text(returns_section.get_text()) if returns_section else '',
                            'type': self.clean_text(return_type_section.get_text()) if return_type_section else ''
                        }
            
            method_info = {
                'method_id': method_id,
                'name': method_name,
                'signature': signature,
                'is_classmethod': is_classmethod,
                'description': description,
                'parameters': parameters,
                'returns': returns if returns else None
            }
            
            return method_info
            
        except Exception as e:
            return None
    
    def parse_attribute(self, attr_dl) -> Optional[Dict[str, Any]]:
        """Parse an attribute/property definition"""
        try:
            attr_dt = attr_dl.find('dt', class_='sig sig-object py')
            if not attr_dt:
                return None
            
            attr_id = attr_dt.get('id', '')
            
            # Extract attribute name
            attr_name_span = attr_dt.find('span', class_='sig-name descname')
            if not attr_name_span:
                return None
            attr_name = self.clean_text(attr_name_span.get_text())
            
            # Extract attribute description
            attr_dd = attr_dl.find('dd')
            description = ''
            attr_type = ''
            enum_values = []
            
            if attr_dd:
                # Get description from first <p> tag
                desc_p = attr_dd.find('p', recursive=False)
                if desc_p:
                    description = self.clean_text(desc_p.get_text())
                
                # Check for enum values (list items before field-list)
                ul_simple = attr_dd.find('ul', class_='simple', recursive=False)
                if ul_simple:
                    for li in ul_simple.find_all('li', recursive=False):
                        enum_text = self.clean_text(li.get_text())
                        if enum_text:
                            enum_values.append(enum_text)
                
                # Extract type information
                field_list = attr_dd.find('dl', class_='field-list')
                if field_list:
                    type_section = self.find_field_section(field_list, 'Type')
                    if type_section:
                        attr_type = self.clean_text(type_section.get_text())
            
            attr_info = {
                'attribute_id': attr_id,
                'name': attr_name,
                'description': description,
                'type': attr_type,
                'enum_values': enum_values if enum_values else None
            }
            
            return attr_info
            
        except Exception as e:
            return None
    
    def parse_parameters(self, param_dd) -> List[Dict[str, Any]]:
        """Parse parameter list from a field section"""
        parameters = []
        
        # Parameters can be in <ul> or directly in <p>
        ul_simple = param_dd.find('ul', class_='simple', recursive=False)
        
        if ul_simple:
            # Multiple parameters in list
            for li in ul_simple.find_all('li', recursive=False):
                param = self.parse_parameter_element(li)
                if param:
                    parameters.append(param)
        else:
            # Single parameter or no <ul>
            for p in param_dd.find_all('p', recursive=False):
                param = self.parse_parameter_element(p)
                if param:
                    parameters.append(param)
        
        return parameters
    
    def parse_parameter_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse a single parameter from a list item or paragraph"""
        try:
            # Find parameter name (in <strong> tag)
            strong = element.find('strong')
            if not strong:
                return None
            
            param_name = self.clean_text(strong.get_text())
            
            # Get full text
            full_text = self.clean_text(element.get_text())
            
            # Remove parameter name from text
            remaining = full_text.replace(param_name, '', 1).strip()
            
            # Split by – or - to separate type and description
            param_type = ''
            param_desc = ''
            
            if '–' in remaining:
                parts = remaining.split('–', 1)
                param_type = parts[0].strip().strip('()')
                param_desc = parts[1].strip() if len(parts) > 1 else ''
            elif ' - ' in remaining:
                parts = remaining.split(' - ', 1)
                param_type = parts[0].strip().strip('()')
                param_desc = parts[1].strip() if len(parts) > 1 else ''
            else:
                # Try to identify type vs description
                if remaining.startswith('('):
                    param_type = remaining.strip().strip('()')
                else:
                    param_desc = remaining
            
            return {
                'name': param_name,
                'type': param_type,
                'description': param_desc
            }
            
        except Exception as e:
            return None
    
    def find_field_section(self, field_list, field_name: str):
        """Find a field section by name (e.g., 'Parameters', 'Returns')"""
        for dt in field_list.find_all('dt', class_='field-odd') + field_list.find_all('dt', class_='field-even'):
            if field_name in dt.get_text():
                dd = dt.find_next_sibling('dd')
                return dd
        return None
    
    def extract_class_name(self, class_dt) -> str:
        """Extract class name from class definition"""
        sig_name = class_dt.find('span', class_='sig-name descname')
        if sig_name:
            return self.clean_text(sig_name.get_text())
        return ''
    
    def extract_base_class(self, class_dt) -> str:
        """Extract base class from class definition"""
        # Look for parameter in signature
        sig_param = class_dt.find('em', class_='sig-param')
        if sig_param:
            n_span = sig_param.find('span', class_='n')
            if n_span:
                return self.clean_text(n_span.get_text())
        return ''
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters"""
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = text.replace('¶', '').strip()
        return text
    
    def save_json(self, data: List[Dict[str, Any]]):
        """Save parsed data to JSON file"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved to: {self.output_file}")
        print(f"Total classes: {len(data)}")
    
    def print_stats(self):
        """Print parsing statistics"""
        print("\n" + "=" * 60)
        print("Parsing Statistics:")
        print("=" * 60)
        print(f"Total files found:     {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"\nTotal classes:         {self.stats['total_classes']}")
        print(f"Total methods:         {self.stats['total_methods']}")
        print(f"Total attributes:      {self.stats['total_attributes']}")
        print("=" * 60)
        
        if self.failed_files:
            print(f"\n⚠️  {len(self.failed_files)} files failed to parse:")
            for item in self.failed_files[:10]:  # Show first 10
                print(f"  - {item['file']}: {item['error']}")
            if len(self.failed_files) > 10:
                print(f"  ... and {len(self.failed_files) - 10} more")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse Blender Python API documentation (bpy.types.* only)'
    )
    parser.add_argument(
        '--docs-dir',
        type=str,
        default='/userhome/cs2/u3665834/projects/hunyuan3D-Agent-G1/bpy_documentation/blender_python_reference_4_5',
        help='HTML documentation directory'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/userhome/cs2/u3665834/projects/hunyuan3D-Agent-G1/bpy_documentation/structured_docs/bpy_types_flat.json',
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    print("Starting Blender Python API Documentation Parser...")
    print(f"Source directory: {args.docs_dir}")
    print(f"Output file: {args.output_file}")
    print()
    
    parser = BpyTypesDocParser(args.docs_dir, args.output_file)
    parser.parse_all()
    
    print("\n✓ Documentation parsing complete!")


if __name__ == '__main__':
    main()
