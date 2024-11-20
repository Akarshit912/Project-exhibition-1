const menuBtn = document.getElementById("menu-btn");
const navLinks = document.getElementById("nav-links");
const menuBtnIcon = menuBtn.querySelector("i");



function changeBackground() {
  const hero = document.querySelector('.Header__image');
  const images = ['img1.jpeg','img2.jpeg']; 
  let currentIndex = 0;

  function updateBackground() {
    // Remove the animation class to reset it
    hero.classList.remove('fadeEffect');

    // Change the background image
    hero.style.backgroundImage = `url('${images[currentIndex]}')`;

    // Force reflow to restart the animation
    void hero.offsetWidth;

    // Re-add the animation class
    hero.classList.add('fadeEffect');

    // Update the current index
    currentIndex = (currentIndex + 1) % images.length;
  }

  // Change background initially
  updateBackground();

  // Set interval to change background every 4 seconds (4000 milliseconds)
  setInterval(updateBackground, 8000);
}

// Call the function when the page loads
window.onload = changeBackground;


menuBtn.addEventListener("click", (e) => {
  navLinks.classList.toggle("open");

  const isOpen = navLinks.classList.contains("open");
  menuBtnIcon.setAttribute("class", isOpen ? "ri-close-line" : "ri-menu-line");
});

navLinks.addEventListener("click", (e) => {
  navLinks.classList.remove("open");
  menuBtnIcon.setAttribute("class", "ri-menu-line");
});

const navSearch = document.getElementById("nav-search");

navSearch.addEventListener("click", (e) => {
  navSearch.classList.toggle("open");
});

const scrollRevealOption = {
  distance: "50px",
  origin: "bottom",
  duration: 1000,
};

ScrollReveal().reveal(".header__image img", {
  ...scrollRevealOption,
  origin: "right",
});
ScrollReveal().reveal(".header__content div", {
  duration: 1000,
  delay: 500,
});
ScrollReveal().reveal(".header__content h1", {
  ...scrollRevealOption,
  delay: 1000,
});
ScrollReveal().reveal(".header__content p", {
  ...scrollRevealOption,
  delay: 1500,
});

ScrollReveal().reveal(".deals__card", {
  ...scrollRevealOption,
  interval: 500,
});

ScrollReveal().reveal(".about__image img", {
  ...scrollRevealOption,
  origin: "right",
});
ScrollReveal().reveal(".about__card", {
  duration: 1000,
  interval: 500,
  delay: 500,
});

const swiper = new swiper(".swiper", {
  loop: true,
});



