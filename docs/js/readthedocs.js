// Trigger the Read the Docs Addons Search modal when clicking on "Search docs" input from the topnav.
// NOTE: The selector of the search input may need to be adjusted based on your theme.
document.querySelector("[role='search'] input").addEventListener("focusin", () => {
    const event = new CustomEvent("readthedocs-search-show");
    document.dispatchEvent(event);
 });