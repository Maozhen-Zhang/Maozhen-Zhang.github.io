export function highlightSearchTerm({ search, selector }) {
  if (!CSS.highlights) {
    return null;
  }

  const highlights = new Highlight();
  const elements = document.querySelectorAll(selector);
  const nonMatchingElements = [];

  elements.forEach((element) => {
    const text = element.innerText.toLowerCase();
    if (text.includes(search)) {
      // Find and highlight the search term
      const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
      let node;
      while (node = walker.nextNode()) {
        const index = node.textContent.toLowerCase().indexOf(search);
        if (index !== -1) {
          const range = new Range();
          range.setStart(node, index);
          range.setEnd(node, index + search.length);
          highlights.add(range);
        }
      }
    } else {
      nonMatchingElements.push(element);
    }
  });

  CSS.highlights.set("search", highlights);
  return nonMatchingElements;
}