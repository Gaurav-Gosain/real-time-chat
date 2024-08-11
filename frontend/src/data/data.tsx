export type UserData = {
  name: string;
};

export const loggedInUserData = {
  id: 5,
  avatar: "/LoggedInUser.jpg",
  name: "Jakob Hoeg",
};

export type LoggedInUserData = typeof loggedInUserData;

export interface Message {
  message: string;
  audio: string;
}

export interface User {
  messages: Message[];
}
