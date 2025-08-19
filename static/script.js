document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);

        const response = await fetch("/", {
            method: "POST",
            body: formData
        });

        const html = await response.text();
        document.body.innerHTML = html; // هيبدل الصفحة بالنتيجة
    });
});
