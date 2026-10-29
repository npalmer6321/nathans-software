(function () {
  var btn = document.querySelector('.menu-btn');
  var nav = document.getElementById('nav');
  btn.addEventListener('click', function () {
    var open = nav.classList.toggle('open');
    btn.setAttribute('aria-expanded', open);
  });
  nav.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') { nav.classList.remove('open'); btn.setAttribute('aria-expanded', false); }
  });

  var amounts = document.querySelectorAll('.amounts button');
  var msg = document.getElementById('donate-msg');
  var link = document.getElementById('donate-link');
  amounts.forEach(function (b) {
    b.addEventListener('click', function () {
      amounts.forEach(function (x) { x.classList.remove('on'); });
      b.classList.add('on');
      var amt = Number(b.dataset.amt);
      msg.textContent = '$' + amt + ' provides about ' + amt * 2 + ' meals.';
      link.textContent = 'Donate $' + amt;
    });
  });

  document.getElementById('year').textContent = new Date().getFullYear();
})();
