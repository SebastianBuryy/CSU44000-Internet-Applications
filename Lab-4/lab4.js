// Write a simple program that counts down from 25 to 0 in decrements of 5 and print each number to the console.
// Now, experimenting with promises, introduce a 5 second delay after printing each number to the console.

async function countdown() {
  let i = 25
  while (i >= 0) {
    console.log(i)
    await new Promise(resolve => setTimeout(resolve, 5000))
    i -= 5
  }
}

countdown()