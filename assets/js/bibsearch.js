import { highlightSearchTerm } from "./highlight-search-term.js";

document.addEventListener("DOMContentLoaded", function () {
  // actual bibsearch logic
  const filterItems = (searchTerm) => {
    document.querySelectorAll(".bibliography, .unloaded").forEach((element) => element.classList.remove("unloaded"));

    // highlight-search-term
    if (CSS.highlights) {
      const nonMatchingElements = highlightSearchTerm({ search: searchTerm, selector: ".bibliography > li" });
      if (nonMatchingElements == null) {
        return;
      }
      nonMatchingElements.forEach((element) => {
        element.classList.add("unloaded");
      });
    } else {
      // Simply add unloaded class to all non-matching items if Browser does not support CSS highlights
      document.querySelectorAll(".bibliography > li").forEach((element, index) => {
        const text = element.innerText.toLowerCase();
        if (text.indexOf(searchTerm) == -1) {
          element.classList.add("unloaded");
        }
      });
    }

    document.querySelectorAll("h2.bibliography").forEach(function (element) {
      const year = element.textContent.trim();
      const items = element.nextElementSibling.querySelectorAll("li:not(.unloaded)");
      if (items.length === 0) {
        element.classList.add("unloaded");
      } else {
        element.classList.remove("unloaded");
      }
    });
  };

  const bibsearchInput = document.getElementById("bibsearch");
  if (bibsearchInput) {
    bibsearchInput.addEventListener("input", function () {
      const searchTerm = this.value.toLowerCase().trim();
      filterItems(searchTerm);
    });
  }
});