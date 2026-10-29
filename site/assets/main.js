(function () {
  var cfg = window.SITE_CONFIG || {};

  var btn = document.querySelector('.menu-btn');
  var nav = document.getElementById('nav');
  btn.addEventListener('click', function () {
    var open = nav.classList.toggle('open');
    btn.setAttribute('aria-expanded', open);
  });
  nav.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') { nav.classList.remove('open'); btn.setAttribute('aria-expanded', false); }
  });

  var donate = document.getElementById('donate-link');
  if (/^https:\/\//.test(cfg.donateUrl || '')) {
    donate.href = cfg.donateUrl;
    donate.textContent = 'Donate';
    donate.removeAttribute('aria-disabled');
    donate.rel = 'noopener';
  } else {
    donate.addEventListener('click', function (e) { e.preventDefault(); });
  }

  var email = cfg.contactEmail || '';
  if (email) document.getElementById('footer-email').href = 'mailto:' + email;

  document.getElementById('volunteer-form').addEventListener('submit', function (e) {
    e.preventDefault();
    if (!email) return;
    var f = e.target;
    var body = 'Name: ' + f.name.value + '\nEmail: ' + f.email.value + '\n\n' + f.message.value;
    window.location.href = 'mailto:' + email +
      '?subject=' + encodeURIComponent("Nathan's Software: " + f.interest.value) +
      '&body=' + encodeURIComponent(body);
  });

  document.getElementById('year').textContent = new Date().getFullYear();
})();
