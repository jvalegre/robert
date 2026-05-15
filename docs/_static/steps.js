document.addEventListener("DOMContentLoaded", () => {
    const steps = document.querySelectorAll(".step-content");

    const tutorials = {};

    steps.forEach(el => {
        const prefix = el.dataset.prefix;
        const step = parseInt(el.dataset.step);

        if (!tutorials[prefix]) {
            tutorials[prefix] = [];
        }

        tutorials[prefix].push(el);
    });

    // Sort steps and initialize
    Object.keys(tutorials).forEach(prefix => {
        tutorials[prefix].sort((a, b) => {
            return a.dataset.step - b.dataset.step;
        });

        tutorials[prefix].forEach(el => el.style.display = "none");

        if (tutorials[prefix][0]) {
            tutorials[prefix][0].style.display = "block";
        }
    });

    // Expose functions globally
    window.nextStep = function(prefix, current) {
        const list = tutorials[prefix];
        if (!list) return;

        list.forEach(el => el.style.display = "none");

        const next = list.find(el => parseInt(el.dataset.step) === current + 1);
        if (next) next.style.display = "block";
    };

    window.prevStep = function(prefix, current) {
        const list = tutorials[prefix];
        if (!list) return;

        list.forEach(el => el.style.display = "none");

        const prev = list.find(el => parseInt(el.dataset.step) === current - 1);
        if (prev) prev.style.display = "block";
    };
});