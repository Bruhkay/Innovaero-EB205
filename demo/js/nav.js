/* Navigation bar shared by all demo pages; marks the current page as active. */
(function () {
  "use strict";
  var pages = [["index.html", "Overview"], ["listen.html", "Listen"], ["fdtd.html", "Cross-section waves"],
               ["quiet_zone.html", "Quiet zone"], ["bay.html", "Skin bay"], ["heatmap.html", "3D heat map"]];
  var here = location.pathname.split("/").pop() || "index.html";
  var nav = document.querySelector("header.top nav");
  if (!nav) return;
  nav.innerHTML = pages.map(function (p) {
    return '<a href="' + p[0] + '"' + (p[0] === here ? ' class="active"' : "") + ">" + p[1] + "</a>";
  }).join("");
})();
