from __future__ import annotations
import os
from typing import Optional, Union, List, Callable, Tuple
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import docx
import PyPDF2
from urllib.parse import urlparse

class FileProcessor:
    """
    A utility class for processing different types of files and returning their contents as strings.
    Supports: DOC, PDF, TXT, CSV, Excel, URL, and HTML files.
    Also supports returning specific chunks of content from files.
    """
    
    @staticmethod
    def read_txt(file_path: Union[str, Path], chunk_size: Optional[int] = None, start_index: int = 0) -> str:
        """
        Read content from a text file.
        
        Args:
            file_path (Union[str, Path]): Path to the text file
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the text file or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                if chunk_size is None:
                    return file.read()
                
                # Move to the start position
                file.seek(start_index)
                return file.read(chunk_size)
        except FileNotFoundError:
            raise FileNotFoundError(f"Text file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")

    @staticmethod
    def read_doc(file_path: Union[str, Path], chunk_size: Optional[int] = None, start_index: int = 0) -> str:
        """
        Read content from a Word document.
        
        Args:
            file_path (Union[str, Path]): Path to the Word document
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the Word document or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            doc = docx.Document(file_path)
            full_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            if chunk_size is None:
                return full_text
            
            # Get the requested chunk
            end_index = min(start_index + chunk_size, len(full_text))
            if start_index >= len(full_text):
                return ""
            
            return full_text[start_index:end_index]
        except FileNotFoundError:
            raise FileNotFoundError(f"Word document not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading Word document: {str(e)}")

    @staticmethod
    def read_pdf(file_path: Union[str, Path], chunk_size: Optional[int] = None, start_index: int = 0) -> str:
        """
        Read content from a PDF file.
        
        Args:
            file_path (Union[str, Path]): Path to the PDF file
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the PDF file or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = '\n'.join([page.extract_text() for page in pdf_reader.pages])
                
                if chunk_size is None:
                    return full_text
                
                # Get the requested chunk
                end_index = min(start_index + chunk_size, len(full_text))
                if start_index >= len(full_text):
                    return ""
                
                return full_text[start_index:end_index]
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    @staticmethod
    def read_csv(
        file_path: Union[str, Path], 
        delimiter: str = ',',
        encoding: str = 'utf-8',
        chunk_size: Optional[int] = None, 
        start_index: int = 0
    ) -> str:
        """
        Read content from a CSV file and return it as a formatted string.
        
        Args:
            file_path (Union[str, Path]): Path to the CSV file
            delimiter (str, optional): CSV delimiter. Defaults to ','
            encoding (str, optional): File encoding. Defaults to 'utf-8'
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the CSV file as a formatted string or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
            full_text = df.to_string()
            
            if chunk_size is None:
                return full_text
            
            # Get the requested chunk
            lines = full_text.split('\n')
            
            # Calculate line-based start and end
            line_start = min(start_index, len(lines))
            line_end = min(line_start + chunk_size, len(lines)) if chunk_size else len(lines)
            
            return '\n'.join(lines[line_start:line_end])
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")

    @staticmethod
    def read_excel(
        file_path: Union[str, Path],
        sheet_name: Optional[Union[str, int]] = 0,
        chunk_size: Optional[int] = None, 
        start_index: int = 0
    ) -> str:
        """
        Read content from an Excel file and return it as a formatted string.
        
        Args:
            file_path (Union[str, Path]): Path to the Excel file
            sheet_name (Optional[Union[str, int]], optional): Sheet to read. Defaults to 0
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the Excel file as a formatted string or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            full_text = df.to_string()
            
            if chunk_size is None:
                return full_text
            
            # Get the requested chunk
            lines = full_text.split('\n')
            
            # Calculate line-based start and end
            line_start = min(start_index, len(lines))
            line_end = min(line_start + chunk_size, len(lines)) if chunk_size else len(lines)
            
            return '\n'.join(lines[line_start:line_end])
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    @staticmethod
    def read_url(url: str, chunk_size: Optional[int] = None, start_index: int = 0, timeout: int = 30) -> str:
        """
        Fetch and read content from a URL.
        
        Args:
            url (str): URL to fetch content from
            timeout (int, optional): Request timeout in seconds. Defaults to 30
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire content. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the webpage with HTML tags removed or chunk of content
            
        Raises:
            ValueError: If the URL is invalid
            Exception: If there's an error fetching the content
        """
        try:
            # Validate URL
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Parse HTML and remove tags
            soup = BeautifulSoup(response.text, 'html.parser')
            full_text = soup.get_text(separator='\n', strip=True)
            
            if chunk_size is None:
                return full_text
            
            # Get the requested chunk
            end_index = min(start_index + chunk_size, len(full_text))
            if start_index >= len(full_text):
                return ""
            
            return full_text[start_index:end_index]
        except ValueError as ve:
            raise ValueError(f"Invalid URL: {str(ve)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching URL content: {str(e)}")

    @staticmethod
    def read_html(file_path: Union[str, Path], chunk_size: Optional[int] = None, start_index: int = 0) -> str:
        """
        Read content from an HTML file and return it as plain text.
        
        Args:
            file_path (Union[str, Path]): Path to the HTML file
            chunk_size (Optional[int], optional): Size of chunk to read. If None, read entire file. Defaults to None.
            start_index (int, optional): Starting index to read from. Defaults to 0.
            
        Returns:
            str: Content of the HTML file with tags removed or chunk of content
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                full_text = soup.get_text(separator='\n', strip=True)
                
                if chunk_size is None:
                    return full_text
                
                # Get the requested chunk
                end_index = min(start_index + chunk_size, len(full_text))
                if start_index >= len(full_text):
                    return ""
                
                return full_text[start_index:end_index]
        except FileNotFoundError:
            raise FileNotFoundError(f"HTML file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error reading HTML file: {str(e)}")

    @staticmethod
    def read_file_chunk(file_path: Union[str, Path], chunk_size: int = 1024, start_index: int = 0) -> str:
        """
        Read a chunk of content from a url or a file based on its type.

        Args:
            file_path (Union[str, Path]): Path to the file or a url
            chunk_size (int, optional): Chunk size in bytes or lines depending on file type. Defaults to 1024. If None, read entire file.
            start_index (int, optional): Starting index to read from. Defaults to 0

        Returns:
            str: Chunk of the file content

        Raises:
            ValueError: If the file type is not supported
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        
        if file_path.startswith('http'):
            return FileProcessor.read_url(file_path, chunk_size=chunk_size, start_index=start_index)
        
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        extension = file_path.suffix.lower()
        
        extension_handlers = {
            '.txt': FileProcessor.read_txt,
            # '.doc': FileProcessor.read_doc,
            # '.docx': FileProcessor.read_doc,
            '.pdf': FileProcessor.read_pdf,
            # '.csv': FileProcessor.read_csv,
            # '.xls': FileProcessor.read_excel,
            # '.xlsx': FileProcessor.read_excel,
            '.html': FileProcessor.read_html,
            '.htm': FileProcessor.read_html
        }
        
        handler = extension_handlers.get(extension)
        if handler is None:
            raise ValueError(f"Unsupported file type: {extension}")
            
        return handler(file_path, chunk_size=chunk_size, start_index=start_index)

    @staticmethod
    def read_file(file_path: Union[str, Path]) -> str:
        """
        Automatically detect file type and read its content.
        
        Args:
            file_path (Union[str, Path]): Path to the file
            
        Returns:
            str: Content of the file
            
        Raises:
            ValueError: If the file type is not supported
            FileNotFoundError: If the file doesn't exist
            Exception: If there's an error reading the file
        """
        return FileProcessor.read_file_chunk(file_path, chunk_size=None, start_index=0)

    @staticmethod
    def get_tools() -> List[Callable]:
        """
        Get a list of available file processing functions.
        
        Returns:
            List[Callable]: List of file processing functions
        """
        return [
            FileProcessor.read_txt,
            # FileProcessor.read_doc,
            FileProcessor.read_pdf,
            # FileProcessor.read_csv,
            # FileProcessor.read_excel,
            FileProcessor.read_url,
            # FileProcessor.read_html
        ]

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """
        Get a list of supported file extensions.
        
        Returns:
            List[str]: List of supported file extensions
        """
        return [
            '.txt', 
            # '.doc', 
            # '.docx', 
            '.pdf', 
            # '.csv', 
            # '.xls', 
            # '.xlsx', 
            '.html', 
            '.htm'
        ]