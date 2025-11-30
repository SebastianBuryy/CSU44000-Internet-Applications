"use strict";

function callmebackin(s, subject) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("Returned call subject: " + subject)
        }, s * 1000)
    })
}

console.log("First I call you about a dog")
let p1 = callmebackin(5, "about a dog")
let p2 = callmebackin(3, "about another dog")
p1.then((subject) => console.log("Got called back: " + subject))
p2.then((subject) => console.log("You called me back: " + subject))